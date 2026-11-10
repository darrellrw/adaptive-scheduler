import os
import threading

import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Scheduler import (
    Benchmark,
    SegmentEncoding,
)

from Scheduler.Information import resolve_share_path

SLAVE_ID = os.environ.get("SLAVE_ID", "worker")
SLAVE_HOST = os.environ.get("SLAVE_HOST", "0.0.0.0")
SLAVE_PORT = int(os.environ.get("SLAVE_PORT", "5000"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/")
SHARE_DIR = os.environ.get("SHARE_DIR", ".")

app = FastAPI(title = "Slave API")

busy_lock = threading.Lock()

class BenchmarkRequest(BaseModel):
    video_path: str
    codec: str = "libx264"
    preset: str = "fast"
    crf: int = 24
    repeat: int = 3
    delay: float = 1

class Segment(BaseModel):
    segment: int
    start: float
    end: float
    duration: float
    start_frame: int
    total_frames: int

class EncodeRequest(BaseModel):
    job_id: str = Field(pattern = r"^[A-Za-z0-9_-]+$")
    video_path: str
    segment: Segment
    codec: str = "libx264"
    preset: str = "fast"
    crf: int = 24

def get_video_path(video_path):
    try:
        full_path = resolve_share_path(video_path)
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    if not os.path.isfile(full_path):
        raise HTTPException(status_code = 404, detail = f"Video {video_path} not found in {SHARE_DIR}.")

    return full_path

def acquire_worker():
    if not busy_lock.acquire(blocking = False):
        raise HTTPException(status_code = 409, detail = f"Worker {SLAVE_ID} is busy.")

@app.get("/health")
def health():
    return {"status": "ok", "worker_id": SLAVE_ID, "busy": busy_lock.locked()}

@app.post("/benchmark")
def benchmark(request: BenchmarkRequest):
    video_path = get_video_path(request.video_path)
    acquire_worker()

    try:
        return Benchmark(video_path).benchmark_encoding(
            codec = request.codec,
            preset = request.preset,
            crf = request.crf,
            repeat = request.repeat,
            delay = request.delay,
        )
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except RuntimeError as e:
        raise HTTPException(status_code = 500, detail = str(e))
    finally:
        busy_lock.release()

@app.post("/encode")
def encode(request: EncodeRequest):
    video_path = get_video_path(request.video_path)
    output_dir = os.path.join(SHARE_DIR, OUTPUT_DIR, request.job_id)
    acquire_worker()

    try:
        result = SegmentEncoding(video_path, output_dir = output_dir).encode_segment(
            request.segment.model_dump(),
            codec = request.codec,
            preset = request.preset,
            crf = request.crf,
        )
    except RuntimeError as e:
        raise HTTPException(status_code = 500, detail = str(e))
    finally:
        busy_lock.release()

    result["worker_id"] = SLAVE_ID
    return result

if __name__ == "__main__":
    uvicorn.run(app, host = SLAVE_HOST, port = SLAVE_PORT)