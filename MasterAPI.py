import os
import csv
import json
import time
import uuid
import logging
import threading
from contextlib import contextmanager
from datetime import datetime

import requests
import uvicorn

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from Scheduler import (
    VideoInformation,
    WorkerRegistry,
    WorkerConflictError,
    AdaptiveDistributor,
    RRStaticDistributor,
    SJFStaticDistributor,
    HeteroStaticDistributor,
    FIFODynamicDistributor,
    SJFDynamicDistributor,
    HeteroDynamicDistributor,
    CombineSegments,
)

from Scheduler.Information import resolve_share_path
from Scheduler.Worker import normalize_file_path

MASTER_HOST = os.environ.get("MASTER_HOST", "0.0.0.0")
MASTER_PORT = int(os.environ.get("MASTER_PORT", "8000"))
ENCODE_TIMEOUT = float(os.environ.get("ENCODE_TIMEOUT", "0")) or None
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3"))
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", "2"))
HEARTBEAT_MISSES = int(os.environ.get("HEARTBEAT_MISSES", "3"))
POLL_INTERVAL = 0.5
BENCHMARK_TIMEOUT = float(os.environ.get("BENCHMARK_TIMEOUT", "1800"))
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
control_lock = threading.Lock()
benchmarking = set()

SCHEDULERS = {
    "adaptive": AdaptiveDistributor,
    "rr_static": RRStaticDistributor,
    "sjf_static": SJFStaticDistributor,
    "hetero_static": HeteroStaticDistributor,
    "fifo_dynamic": FIFODynamicDistributor,
    "sjf_dynamic": SJFDynamicDistributor,
    "hetero_dynamic": HeteroDynamicDistributor,
}

def get_scheduler_class(name):
    scheduler_class = SCHEDULERS.get(name)

    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {sorted(SCHEDULERS)}")

    return scheduler_class

class WorkerCreate(BaseModel):
    id: str = Field(min_length = 1, max_length = 64, pattern = r"^[A-Za-z0-9_-]+$")
    address: str = Field(min_length = 1)
    port: int = Field(ge = 1, le = 65535)

class BenchmarkRequest(BaseModel):
    video_path: str = Field(min_length = 1)
    codec: str = "libx264"
    preset: str = "fast"
    crf: int = 24
    repeat: int = Field(3, ge = 3)
    delay: float = Field(1, ge = 0)

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

def snapshot_worker_benchmarks(workers, distributor, codec, preset, crf):
    snapshot = {}

    for worker_id, worker in workers.items():
        state = distributor.state.get(worker_id, {})
        per_file = [
            b for b in worker.list_benchmarks()
            if b["codec"] == codec and b["preset"] == preset and str(b["crf"]) == str(crf)
        ]

        snapshot[worker_id] = {
            "address": worker.address,
            "port": worker.port,
            "setting": {"codec": codec, "preset": preset, "crf": crf},
            "merged": worker.get_benchmark(codec, preset, crf)["benchmark_summary"] if worker.has_benchmark(codec, preset, crf) else None,
            "per_file": per_file,
            "initial_planning": {
                "avg_fps": state.get("avg_fps"),
                "stddev_fps": state.get("stddev_fps"),
                "planning_fps": state.get("planning_fps"),
                "sample_count": state.get("sample_count"),
            },
            "all_benchmarks": worker.list_benchmarks(),
        }

    return snapshot

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
        self.failures = []
        self.last_seen = {worker_id: 0.0 for worker_id in workers}
        self.stop_event = threading.Event()
        self.worker_benchmarks = snapshot_worker_benchmarks(workers, distributor, request.codec, request.preset, request.crf)

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
        timeout = (HEALTH_TIMEOUT, ENCODE_TIMEOUT),
    )

    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    result = response.json()

    if result.get("encoding_time", 0) <= 0:
        raise RuntimeError(f"Invalid encoding_time in response: {result}")

    return result

class EncodeCall:
    def __init__(self, worker, job, segment):
        self.result = None
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target = self.run, args = (worker, job, segment), daemon = True)

    def run(self, worker, job, segment):
        try:
            self.result = send_encode(worker, job, segment)
        except Exception as e:
            self.error = e
        finally:
            self.done.set()

    def start(self):
        self.thread.start()
        return self

def mark_worker_lost(job, worker_id, method, reason):
    with job.cond:
        distributor = job.distributor

        if not distributor.state[worker_id]["active"]:
            return False

        detected_at = job.elapsed()
        last_seen = job.last_seen.get(worker_id, 0.0)
        open_records = [r for r in job.timeline if r["worker_id"] == worker_id and r["end"] is None]

        for record in open_records:
            job.end_record(record, "worker_lost", error = f"{method}: {reason}")

        returned = distributor.disable_worker(worker_id)

        failure = {
            "worker_id": worker_id,
            "method": method,
            "reason": reason,
            "last_seen": round(last_seen, 3),
            "detected_at": round(detected_at, 3),
            "detection_latency": round(detected_at - last_seen, 3),
            "returned_segments": [s["segment"] for s in returned],
            "lost_frames": sum(r["total_frames"] for r in open_records),
            "lost_work_time": round(sum(r["wall_time"] for r in open_records), 3),
        }
        job.failures.append(failure)
        job.log(
            f"LOST   {worker_id} via {method} ({reason}), terakhir hidup t={last_seen:.2f}s, "
            f"latensi deteksi {failure['detection_latency']:.2f}s, segmen dikembalikan: {failure['returned_segments']}"
        )

        if not distributor.get_active_workers():
            job.error = "Semua worker tidak terjangkau."

        job.cond.notify_all()
        return True

def heartbeat_loop(job, worker_id):
    worker = job.workers[worker_id]
    distributor = job.distributor
    misses = 0

    while not job.stop_event.is_set():
        with job.cond:
            if not distributor.state[worker_id]["active"] or distributor.is_finished():
                return

        health = ping_worker(worker.address, worker.port)

        if health.get("status") == "ok":
            misses = 0

            with job.cond:
                job.last_seen[worker_id] = job.elapsed()
        else:
            misses += 1

            if misses >= HEARTBEAT_MISSES:
                mark_worker_lost(job, worker_id, "heartbeat", f"{misses}x ping gagal ({health.get('status')})")
                return

        job.stop_event.wait(HEARTBEAT_INTERVAL)

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

        call = EncodeCall(worker, job, segment).start()

        while not call.done.wait(timeout = POLL_INTERVAL):
            with job.cond:
                if not distributor.state[worker_id]["active"]:
                    # heartbeat sudah menyatakan worker hilang, hasil request (jika datang) diabaikan
                    return

        with job.cond:
            if not distributor.state[worker_id]["active"]:
                return

            error = call.error

            if isinstance(error, (requests.ConnectionError, requests.Timeout)):
                mark_worker_lost(job, worker_id, "connection", type(error).__name__)
                return

            if error is not None:
                job.end_record(record, "failed", error = str(error)[:300])
                accepted = distributor.fail_segment(worker_id, segment)
                job.log(f"FAILED segmen {segment['segment']} di {worker_id} setelah {record['wall_time']:.2f}s ({'dicoba ulang' if accepted else 'menyerah'}): {error}")
                job.cond.notify_all()
                continue

            result = call.result
            job.last_seen[worker_id] = job.elapsed()
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
        "worker_benchmarks": job.worker_benchmarks,
        "merge": job.merge,
        "fault_detection": {
            "heartbeat_interval": HEARTBEAT_INTERVAL,
            "heartbeat_misses": HEARTBEAT_MISSES,
            "health_timeout": HEALTH_TIMEOUT,
            "encode_timeout": ENCODE_TIMEOUT,
        },
        "failures": job.failures,
    }

def run_job(job):
    worker_ids = job.distributor.get_active_workers()

    with job.cond:
        start = job.elapsed()

        for worker_id in worker_ids:
            job.last_seen[worker_id] = start

    threads = [
        threading.Thread(target = worker_loop, args = (job, worker_id), daemon = True)
        for worker_id in worker_ids
    ]
    heartbeats = [
        threading.Thread(target = heartbeat_loop, args = (job, worker_id), daemon = True)
        for worker_id in worker_ids
    ]

    for thread in threads + heartbeats:
        thread.start()

    for thread in threads:
        thread.join()

    job.stop_event.set()

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
            job_dir = os.path.join(RESULT_DIR, job.job_id)
            files = [os.path.join(job_dir, job.results[key]["output_file"]) for key in sorted(job.results)]
            output_path = CombineSegments(output_dir = job_dir).combine(files, job.video_path, job.job_id)
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
            "failures": job.failures,
        }

def get_worker_or_404(worker_id):
    try:
        return registry.get_worker(worker_id)
    except ValueError as e:
        raise HTTPException(status_code = 404, detail = str(e))

def check_video_path(video_path):
    try:
        full_path = resolve_share_path(video_path)
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    if not os.path.isfile(full_path):
        raise HTTPException(status_code = 404, detail = f"Video {video_path} not found in {SHARE_DIR}.")

    return full_path

def ping_worker(address, port):
    try:
        response = requests.get(f"http://{address}:{port}/health", timeout = HEALTH_TIMEOUT)
        return response.json() if response.ok else {"status": f"HTTP {response.status_code}"}
    except (requests.RequestException, ValueError):
        return {"status": "unreachable"}

@contextmanager
def benchmark_session(worker_ids):
    worker_ids = set(worker_ids)

    with control_lock:
        if get_active_job():
            raise HTTPException(status_code = 409, detail = "A job is running, benchmark would disturb the measurement.")

        busy = sorted(worker_ids & benchmarking)

        if busy:
            raise HTTPException(status_code = 409, detail = f"Benchmark already running on worker(s): {busy}")

        benchmarking.update(worker_ids)

    try:
        yield
    finally:
        with control_lock:
            benchmarking.difference_update(worker_ids)

def run_benchmark(worker_id, req):
    try:
        return registry.benchmark_worker(
            worker_id,
            req.video_path,
            codec = req.codec,
            preset = req.preset,
            crf = req.crf,
            repeat = req.repeat,
            delay = req.delay,
            timeout = (HEALTH_TIMEOUT, BENCHMARK_TIMEOUT),
        )
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except requests.HTTPError as e:
        raise HTTPException(status_code = 502, detail = f"Worker {worker_id} responded {e.response.status_code}: {e.response.text[:300]}")
    except requests.Timeout:
        raise HTTPException(status_code = 504, detail = f"Worker {worker_id} timed out (limit {BENCHMARK_TIMEOUT}s).")
    except requests.RequestException as e:
        raise HTTPException(status_code = 502, detail = f"Worker {worker_id} unreachable: {e}")

def benchmark_key(req):
    return (normalize_file_path(req.video_path), req.codec, req.preset, str(req.crf))

def summarize_benchmark(result):
    return {"setting": result["setting"], "benchmark_summary": result["benchmark_summary"]}

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
            item["health"] = ping_worker(worker.address, worker.port)

        result.append(item)

    return result

@app.post("/workers")
def add_worker(request: WorkerCreate):
    try:
        worker = registry.add_worker(request.id, request.address, request.port)
    except ValueError as e:
        raise HTTPException(status_code = 409, detail = str(e))

    return {"id": worker.id, "address": worker.address, "port": worker.port}

@app.post("/workers/batch", status_code = 201)
def add_workers_batch(request: list[WorkerCreate], check_health: bool = False):
    if not request:
        raise HTTPException(status_code = 422, detail = "Request body must contain at least one worker.")

    if check_health:
        unhealthy = []

        for w in request:
            health = ping_worker(w.address, w.port)

            if health.get("status") != "ok":
                unhealthy.append({"id": w.id, "address": w.address, "port": w.port, "health": health})

        if unhealthy:
            raise HTTPException(status_code = 502, detail = {
                "message": "Some workers are not healthy, nothing was added.",
                "workers": unhealthy,
            })

    try:
        workers = registry.add_workers([w.model_dump() for w in request])
    except WorkerConflictError as e:
        raise HTTPException(status_code = 409, detail = {
            "message": f"{e}. Nothing was added.",
            "duplicates": e.duplicates,
            "existing": e.existing,
        })

    return {
        "total": len(workers),
        "added": [{"id": w.id, "address": w.address, "port": w.port} for w in workers],
    }

@app.delete("/workers/{worker_id}")
def remove_worker(worker_id: str):
    try:
        registry.remove_worker(worker_id)
    except ValueError as e:
        raise HTTPException(status_code = 404, detail = str(e))

    return {"removed": worker_id}

@app.post("/workers/{worker_id}/benchmark")
def benchmark_worker(worker_id: str, request: BenchmarkRequest):
    get_worker_or_404(worker_id)
    check_video_path(request.video_path)

    with benchmark_session([worker_id]):
        return run_benchmark(worker_id, request)

@app.post("/workers/{worker_id}/benchmark/batch")
def benchmark_worker_batch(worker_id: str, request: list[BenchmarkRequest], stop_on_error: bool = False, cooldown: float = Query(0, ge = 0), detail: bool = False):
    if not request:
        raise HTTPException(status_code = 422, detail = "Request body must contain at least one benchmark item.")

    get_worker_or_404(worker_id)

    errors = []
    seen = {}

    for index, req in enumerate(request):
        try:
            check_video_path(req.video_path)
        except HTTPException as e:
            errors.append({"index": index, "video_path": req.video_path, "error": e.detail})
            continue

        key = benchmark_key(req)

        if key in seen:
            errors.append({"index": index, "video_path": req.video_path, "error": f"Duplicate of item {seen[key]} (same file, codec, preset, crf)."})
        else:
            seen[key] = index

    if errors:
        raise HTTPException(status_code = 400, detail = {
            "message": "Invalid batch, nothing was benchmarked.",
            "errors": errors,
        })

    items = []
    stopped = False

    with benchmark_session([worker_id]):
        for index, req in enumerate(request):
            item = {
                "index": index,
                "video_path": req.video_path,
                "codec": req.codec,
                "preset": req.preset,
                "crf": req.crf,
            }

            if stopped:
                item["status"] = "skipped"
                items.append(item)
                continue

            if index > 0 and cooldown > 0:
                time.sleep(cooldown)

            try:
                result = run_benchmark(worker_id, req)
                item["status"] = "ok"
                item["result"] = result if detail else summarize_benchmark(result)
            except HTTPException as e:
                item["status"] = "error"
                item["status_code"] = e.status_code
                item["error"] = e.detail
                stopped = stop_on_error

            items.append(item)

    counts = {s: sum(1 for i in items if i["status"] == s) for s in ("ok", "error", "skipped")}
    response = {
        "worker_id": worker_id,
        "total": len(items),
        "succeeded": counts["ok"],
        "failed": counts["error"],
        "skipped": counts["skipped"],
        "items": items,
    }

    if counts["ok"] == 0:
        raise HTTPException(status_code = 502, detail = response)

    return response

@app.post("/jobs")
def create_job(request: JobCreate):
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

    with control_lock:
        if get_active_job():
            raise HTTPException(status_code = 409, detail = "Another job is still running.")

        if benchmarking:
            raise HTTPException(status_code = 409, detail = f"Benchmark is running on worker(s) {sorted(benchmarking)}, wait until it finishes.")

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