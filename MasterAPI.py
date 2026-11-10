import os
import csv
import json
import time
import uuid
import logging
import threading
from datetime import datetime

import requests
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Scheduler import (
    VideoInformation,
    WorkerRegistry,
    AdaptiveDistributor,
    FIFODistributor,
    SJFDistributor,
    HeterogeneousDistributor,
    CombineSegments,
)

from Scheduler.Information import resolve_share_path

MASTER_HOST = os.environ.get("MASTER_HOST", "0.0.0.0")
MASTER_PORT = int(os.environ.get("MASTER_PORT", "8000"))
ENCODE_TIMEOUT = float(os.environ.get("ENCODE_TIMEOUT", "600"))
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3"))
MASTER_LOG_DIR = os.environ.get("MASTER_LOG_DIR", "logs/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/")
SHARE_DIR = os.environ.get("SHARE_DIR", ".")

RESULT_DIR = os.path.join(SHARE_DIR, OUTPUT_DIR)

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("master")

app = FastAPI(title = "Master API")

registry = WorkerRegistry()
jobs = {}
jobs_lock = threading.Lock()

SCHEDULERS = {
    "adaptive": AdaptiveDistributor,
    "fifo": FIFODistributor,
    "sjf": SJFDistributor,
    "heterogeneous": HeterogeneousDistributor,
}

def get_scheduler_class(name):
    scheduler_class = SCHEDULERS.get(name)

    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {sorted(SCHEDULERS)}")

    return scheduler_class

class WorkerCreate(BaseModel):
    id: str
    address: str
    port: int

class BenchmarkRequest(BaseModel):
    video_path: str
    codec: str = "libx264"
    preset: str = "fast"
    crf: int = 24
    repeat: int = 3
    delay: float = 1

class JobCreate(BaseModel):
    video_path: str
    scheduler: str = "adaptive"
    codec: str = "libx264"
    preset: str = "fast"
    crf: int = 24
    initial_size: int = 1
    max_retries: int = 3

TIMELINE_FIELDS = [
    "segment", "attempt", "worker_id", "total_frames", "status",
    "start", "end", "wall_time", "encoding_time", "overhead", "encoding_fps",
    "output_file", "error",
]

def new_record(segment, worker_id, attempt, start):
    return {
        "segment": segment["segment"],
        "attempt": attempt,
        "worker_id": worker_id,
        "total_frames": segment["total_frames"],
        "status": "running",
        "start": round(start, 3),
        "end": None,
        "wall_time": None,
        "encoding_time": None,
        "overhead": None,
        "encoding_fps": None,
        "output_file": None,
        "error": None,
    }

def close_record(record, end, status, encoding_time = None, output_file = None, error = None):
    record["end"] = round(end, 3)
    record["wall_time"] = round(end - record["start"], 3)
    record["status"] = status
    record["error"] = error

    if encoding_time is not None:
        record["encoding_time"] = round(encoding_time, 3)
        record["encoding_fps"] = round(record["total_frames"] / encoding_time, 2)
        record["overhead"] = round(max(0.0, record["wall_time"] - encoding_time), 3)

    if output_file is not None:
        record["output_file"] = output_file

def summarize_timeline(timeline):
    finished = [r for r in timeline if r["end"] is not None]
    completed = [r for r in finished if r["status"] == "completed"]
    makespan = max((r["end"] for r in finished), default = 0.0)
    workers = {}

    for r in finished:
        w = workers.setdefault(r["worker_id"], {
            "segments": 0,
            "frames": 0,
            "failed_attempts": 0,
            "busy_time": 0.0,
            "first_start": r["start"],
            "last_end": r["end"],
        })

        if r["status"] == "completed":
            w["segments"] += 1
            w["frames"] += r["total_frames"]
        else:
            w["failed_attempts"] += 1

        w["busy_time"] += r["end"] - r["start"]
        w["first_start"] = min(w["first_start"], r["start"])
        w["last_end"] = max(w["last_end"], r["end"])

    for w in workers.values():
        w["busy_time"] = round(w["busy_time"], 3)
        w["idle_time"] = round(max(0.0, makespan - w["busy_time"]), 3)
        w["utilization"] = round(w["busy_time"] / makespan, 4) if makespan > 0 else 0.0

    frames = sum(r["total_frames"] for r in completed)

    return {
        "makespan": round(makespan, 3),
        "attempts": len(finished),
        "completed_segments": len(completed),
        "frames_encoded": frames,
        "overall_fps": round(frames / makespan, 2) if makespan > 0 else 0.0,
        "workers": workers,
    }

def write_log_files(name, payload):
    try:
        os.makedirs(MASTER_LOG_DIR, exist_ok = True)
        json_path = os.path.join(MASTER_LOG_DIR, f"{name}.json")
        csv_path = os.path.join(MASTER_LOG_DIR, f"{name}.csv")

        with open(json_path, "w", encoding = "utf-8") as f:
            json.dump(payload, f, indent = 2, default = str)

        with open(csv_path, "w", newline = "", encoding = "utf-8") as f:
            writer = csv.DictWriter(f, fieldnames = TIMELINE_FIELDS, extrasaction = "ignore")
            writer.writeheader()
            writer.writerows(payload["timeline"])

        return {"json": json_path, "csv": csv_path}
    except OSError as e:
        logger.error("Gagal menulis log %s: %s", name, e)
        return {}

class Job:
    def __init__(self, job_id, request, distributor, workers):
        self.job_id = job_id
        self.video_path = request.video_path
        self.scheduler = request.scheduler
        self.codec = request.codec
        self.preset = request.preset
        self.crf = request.crf
        self.distributor = distributor
        self.workers = workers
        self.status = "running"
        self.error = None
        self.results = {}
        self.events = []
        self.timeline = []
        self.log_files = {}
        self.created_at = time.time()
        self.finished_at = None
        self.cond = threading.Condition()
        self.load_video_time = 0.0
        self.merge = None

    def elapsed(self):
        return time.time() - self.created_at

    def log(self, message):
        line = f"[{self.elapsed():7.2f}s] {message}"
        self.events.append(line)
        del self.events[:-100]
        logger.info("[job %s] %s", self.job_id, line)

    def begin_record(self, worker_id, segment):
        attempt = sum(1 for r in self.timeline if r["segment"] == segment["segment"]) + 1
        record = new_record(segment, worker_id, attempt, self.elapsed())
        self.timeline.append(record)
        return record

    def end_record(self, record, status, result = None, error = None):
        close_record(
            record,
            self.elapsed(),
            status,
            encoding_time = result["encoding_time"] if result else None,
            output_file = result.get("output_file") if result else None,
            error = error,
        )

def get_active_job():
    with jobs_lock:
        for job in jobs.values():
            if job.status in ("running", "merging"):
                return job

    return None

def send_encode(worker, job, segment):
    response = requests.post(
        f"http://{worker.address}:{worker.port}/encode",
        json = {
            "job_id": job.job_id,
            "video_path": job.video_path,
            "segment": segment,
            "codec": job.codec,
            "preset": job.preset,
            "crf": job.crf,
        },
        timeout = ENCODE_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    result = response.json()

    if result.get("encoding_time", 0) <= 0:
        raise RuntimeError(f"Invalid encoding_time in response: {result}")

    return result

def worker_loop(job, worker_id):
    worker = job.workers[worker_id]
    distributor = job.distributor

    while True:
        with job.cond:
            segment = None

            while segment is None:
                if not distributor.state[worker_id]["active"] or distributor.is_finished():
                    return

                running = distributor.get_running_segments(worker_id)
                segment = running[0] if running else distributor.assign_next(worker_id)

                if segment is None:
                    job.cond.wait(timeout = 1.0)

            record = job.begin_record(worker_id, segment)
            job.log(f"START  segmen {segment['segment']} (percobaan {record['attempt']}, {segment['total_frames']} frame) -> {worker_id}")

        try:
            result = send_encode(worker, job, segment)
        except (requests.ConnectionError, requests.Timeout) as e:
            with job.cond:
                job.end_record(record, "worker_lost", error = type(e).__name__)
                returned = distributor.disable_worker(worker_id)
                job.log(f"LOST   {worker_id} tidak terjangkau ({type(e).__name__}) setelah {record['wall_time']:.2f}s, segmen dikembalikan: {[s['segment'] for s in returned]}")

                if not distributor.get_active_workers():
                    job.error = "Semua worker tidak terjangkau."

                job.cond.notify_all()

            return
        except Exception as e:
            with job.cond:
                job.end_record(record, "failed", error = str(e)[:300])
                accepted = distributor.fail_segment(worker_id, segment)
                job.log(f"FAILED segmen {segment['segment']} di {worker_id} setelah {record['wall_time']:.2f}s ({'dicoba ulang' if accepted else 'menyerah'}): {e}")
                job.cond.notify_all()

            continue

        with job.cond:
            job.end_record(record, "completed", result = result)
            distributor.complete_segment(worker_id, segment, result["encoding_time"])
            job.results[segment["segment"]] = {
                "segment": segment["segment"],
                "worker_id": worker_id,
                "total_frames": segment["total_frames"],
                "encoding_time": result["encoding_time"],
                "encoding_fps": result["encoding_fps"],
                "output_file": result["output_file"],
            }
            job.log(
                f"DONE   segmen {segment['segment']} di {worker_id}: "
                f"t={record['start']:.2f}s -> {record['end']:.2f}s, wall {record['wall_time']:.2f}s, "
                f"encode {result['encoding_time']:.2f}s ({result['encoding_fps']:.1f} fps), overhead {record['overhead']:.2f}s"
            )
            job.cond.notify_all()

def build_job_log(job):
    return {
        "mode": "real",
        "job_id": job.job_id,
        "scheduler": job.scheduler,
        "video_path": job.video_path,
        "codec": job.codec,
        "preset": job.preset,
        "crf": job.crf,
        "status": job.status,
        "error": job.error,
        "created_at": datetime.fromtimestamp(job.created_at).isoformat(timespec = "seconds"),
        "elapsed": round((job.finished_at or time.time()) - job.created_at, 3),
        "summary": summarize_timeline(job.timeline),
        "scheduler_status": job.distributor.get_status(),
        "timeline": job.timeline,
        "events": job.events,
        "load_video_time": job.load_video_time,
        "workers": list(job.workers),
        "merge": job.merge,
    }

def run_job(job):
    threads = [
        threading.Thread(target = worker_loop, args = (job, worker_id), daemon = True)
        for worker_id in job.distributor.get_active_workers()
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    with job.cond:
        status = job.distributor.get_status()

        if status["finished"] and status["failed"] == 0:
            job.status = "merging"
        elif status["finished"]:
            job.status = "completed_with_failures"
        else:
            job.status = "failed"
            job.error = job.error or "Job berhenti sebelum semua segmen selesai."

    if job.status == "merging":
        merge_start = job.elapsed()
        job.log("MERGE  menggabungkan segmen tanpa re-encode")

        try:
            files = [os.path.join(RESULT_DIR, job.job_id, job.results[key]["output_file"]) for key in sorted(job.results)]
            output_path = CombineSegments(output_dir = RESULT_DIR).combine(files, job.video_path, job.job_id)
            merge_error = None
            final_status = "completed"
        except Exception as e:
            output_path = None
            merge_error = str(e)[:300]
            final_status = "failed"
            job.error = f"Gagal menggabungkan segmen: {merge_error}"

        merge_end = job.elapsed()
        job.merge = {
            "start": round(merge_start, 3),
            "end": round(merge_end, 3),
            "duration": round(merge_end - merge_start, 3),
            "output_file": output_path,
            "error": merge_error,
        }
        job.log(f"MERGE  {final_status} dalam {job.merge['duration']:.2f}s -> {output_path or merge_error}")
    else:
        final_status = job.status

    with job.cond:
        job.status = final_status
        job.finished_at = time.time()
        summary = summarize_timeline(job.timeline)
        job.log(f"job {job.status}: makespan {summary['makespan']:.2f}s, {summary['completed_segments']} segmen selesai, {summary['attempts']} percobaan")
        job.log_files = write_log_files(f"job_{job.job_id}", build_job_log(job))

def get_job_summary(job, event_count = 20):
    with job.cond:
        end_time = job.finished_at or time.time()

        return {
            "job_id": job.job_id,
            "scheduler": job.scheduler,
            "status": job.status,
            "error": job.error,
            "video_path": job.video_path,
            "codec": job.codec,
            "preset": job.preset,
            "crf": job.crf,
            "elapsed": end_time - job.created_at,
            "scheduler_status": job.distributor.get_status(),
            "timeline_summary": summarize_timeline(job.timeline),
            "log_files": job.log_files,
            "results": [job.results[key] for key in sorted(job.results)],
            "events": job.events[-event_count:],
            "merge": job.merge,
        }

@app.get("/schedulers")
def list_schedulers():
    return {"schedulers": sorted(SCHEDULERS)}

@app.get("/workers")
def list_workers(check_health: bool = False):
    result = []

    for worker in registry.list_workers():
        item = {
            "id": worker.id,
            "address": worker.address,
            "port": worker.port,
            "benchmarks": worker.list_benchmarks(),
            "merged_benchmarks": worker.list_merged_benchmarks(),
        }

        if check_health:
            try:
                response = requests.get(f"http://{worker.address}:{worker.port}/health", timeout = HEALTH_TIMEOUT)
                item["health"] = response.json() if response.ok else {"status": f"HTTP {response.status_code}"}
            except requests.RequestException:
                item["health"] = {"status": "unreachable"}

        result.append(item)

    return result

@app.post("/workers")
def add_worker(request: WorkerCreate):
    try:
        worker = registry.add_worker(request.id, request.address, request.port)
    except ValueError as e:
        raise HTTPException(status_code = 409, detail = str(e))

    return {"id": worker.id, "address": worker.address, "port": worker.port}

@app.delete("/workers/{worker_id}")
def remove_worker(worker_id: str):
    try:
        registry.remove_worker(worker_id)
    except ValueError as e:
        raise HTTPException(status_code = 404, detail = str(e))

    return {"removed": worker_id}

@app.post("/workers/{worker_id}/benchmark")
def benchmark_worker(worker_id: str, request: BenchmarkRequest):
    if get_active_job():
        raise HTTPException(status_code = 409, detail = "A job is running, benchmark would disturb the measurement.")

    try:
        registry.get_worker(worker_id)
        return registry.benchmark_worker(
            worker_id,
            request.video_path,
            codec = request.codec,
            preset = request.preset,
            crf = request.crf,
            repeat = request.repeat,
            delay = request.delay,
        )
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except requests.HTTPError as e:
        raise HTTPException(status_code = 502, detail = f"Worker {worker_id} responded {e.response.status_code}: {e.response.text[:300]}")
    except requests.RequestException as e:
        raise HTTPException(status_code = 502, detail = f"Worker {worker_id} unreachable: {e}")

@app.post("/workers/{worker_id}/benchmark/batch")
def benchmark_worker_batch(worker_id: str, request: list[BenchmarkRequest]):
    if get_active_job():
        raise HTTPException(status_code = 409, detail = "A job is running, benchmark would disturb the measurement.")

    results = []

    try:
        for req in request:
            registry.get_worker(worker_id)
            results.append(registry.benchmark_worker(
                worker_id,
                req.video_path,
                codec = req.codec,
                preset = req.preset,
                crf = req.crf,
                repeat = req.repeat,
                delay = req.delay,
            ))
        
        return results
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except requests.HTTPError as e:
        raise HTTPException(status_code = 502, detail = f"Worker {worker_id} responded {e.response.status_code}: {e.response.text[:300]}")
    except requests.RequestException as e:
        raise HTTPException(status_code = 502, detail = f"Worker {worker_id} unreachable: {e}")

@app.post("/jobs")
def create_job(request: JobCreate):
    if get_active_job():
        raise HTTPException(status_code = 409, detail = "Another job is still running.")

    try:
        scheduler_class = get_scheduler_class(request.scheduler)
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    try:
        video_path = resolve_share_path(request.video_path)
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    if not os.path.isfile(video_path):
        raise HTTPException(status_code = 404, detail = f"Video {request.video_path} not found in {SHARE_DIR}.")

    try:
        load_start = time.time()
        video_information = VideoInformation(video_path).get_video_information()
        load_time = time.time() - load_start
        distributor = scheduler_class(
            registry,
            video_information,
            codec = request.codec,
            preset = request.preset,
            crf = request.crf,
            max_retries = request.max_retries,
        )
        job = Job(uuid.uuid4().hex[:8], request, distributor, {worker.id: worker for worker in registry.list_workers()})
        job.load_video_time = round(load_time, 3)

        with job.cond:
            distributor.initial_assignment(initial_size = request.initial_size)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code = 400, detail = str(e))

    with jobs_lock:
        jobs[job.job_id] = job

    job.log(f"job dibuat: scheduler {job.scheduler}, {len(distributor.segments)} segmen, {len(job.workers)} worker")
    threading.Thread(target = run_job, args = (job,), daemon = True).start()

    return {"job_id": job.job_id, "scheduler": job.scheduler, "status": job.status, "total_segments": len(distributor.segments)}

@app.get("/jobs")
def list_jobs():
    with jobs_lock:
        items = list(jobs.values())

    return [{"job_id": job.job_id, "scheduler": job.scheduler, "status": job.status, "video_path": job.video_path} for job in items]

@app.get("/jobs/{job_id}")
def get_job(job_id: str, events: int = 20):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code = 404, detail = f"Job {job_id} not found.")

    return get_job_summary(job, events)

@app.get("/jobs/{job_id}/log")
def get_job_log(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code = 404, detail = f"Job {job_id} not found.")

    with job.cond:
        payload = build_job_log(job)

    payload["log_files"] = job.log_files
    return payload

if __name__ == "__main__":
    uvicorn.run(app, host = MASTER_HOST, port = MASTER_PORT)
