import os
import math
import threading
from collections import Counter

import requests

def normalize_file_path(file_path):
    if file_path is None:
        return None

    return os.path.normpath(str(file_path)).replace("\\", "/")

def merge_summaries(summaries):
    entries = [
        (
            int(s["total_runs"]),
            float(s["average_encoding_fps"]),
            float(s["stddev_encoding_fps"]),
        )
        for s in summaries
        if int(s["total_runs"]) > 0
    ]

    total_runs = sum(n for n, _, _ in entries)

    if total_runs == 0:
        raise ValueError("No benchmark runs to merge.")

    mean = sum(n * m for n, m, _ in entries) / total_runs

    if total_runs > 1:
        m2 = sum((n - 1) * (sd ** 2) + n * ((m - mean) ** 2) for n, m, sd in entries)
        stddev = math.sqrt(m2 / (total_runs - 1))
    else:
        stddev = 0.0

    return {
        "average_encoding_fps": mean,
        "stddev_encoding_fps": stddev,
        "total_runs": total_runs,
        "total_files": len(entries),
    }

class WorkerProfile:
    def __init__(self, id, address, port):
        self.id = id
        self.address = address
        self.port = port
        self.benchmark = {}

    def add_benchmark(self, benchmark):
        setting = benchmark["setting"]
        key = (
            normalize_file_path(setting.get("file_path")),
            setting["codec"],
            setting["preset"],
            str(setting["crf"]),
        )
        self.benchmark[key] = benchmark

    def get_matching_benchmarks(self, codec, preset, crf, file_path = None):
        setting = (codec, preset, str(crf))
        file_path = normalize_file_path(file_path)

        return {
            key: benchmark
            for key, benchmark in self.benchmark.items()
            if key[1:] == setting and (file_path is None or key[0] == file_path)
        }

    def get_benchmark(self, codec, preset, crf, file_path = None):
        matches = self.get_matching_benchmarks(codec, preset, crf, file_path)

        if not matches:
            where = f", file_path: {file_path}" if file_path is not None else ""
            raise ValueError(f"No benchmark found for codec: {codec}, preset: {preset}, crf: {crf}{where}")

        return {
            "setting": {
                "codec": codec,
                "preset": preset,
                "crf": crf,
                "file_paths": sorted(key[0] for key in matches if key[0] is not None),
            },
            "benchmark_summary": merge_summaries([b["benchmark_summary"] for b in matches.values()]),
        }

    def has_benchmark(self, codec, preset, crf, file_path = None):
        return bool(self.get_matching_benchmarks(codec, preset, crf, file_path))

    def get_fps(self, codec, preset, crf, file_path = None):
        return self.get_benchmark(codec, preset, crf, file_path)["benchmark_summary"]["average_encoding_fps"]

    def get_fps_stddev(self, codec, preset, crf, file_path = None):
        return self.get_benchmark(codec, preset, crf, file_path)["benchmark_summary"]["stddev_encoding_fps"]

    def get_sample_count(self, codec, preset, crf, file_path = None):
        return self.get_benchmark(codec, preset, crf, file_path)["benchmark_summary"]["total_runs"]

    def get_coefficient_of_variation(self, codec, preset, crf, file_path = None):
        summary = self.get_benchmark(codec, preset, crf, file_path)["benchmark_summary"]
        avg_fps = summary["average_encoding_fps"]
        return summary["stddev_encoding_fps"] / avg_fps if avg_fps != 0 else float("inf")

    def get_estimated_time(self, codec, preset, crf, total_frames, file_path = None):
        summary = self.get_benchmark(codec, preset, crf, file_path)["benchmark_summary"]
        planning_fps = summary["average_encoding_fps"] - summary["stddev_encoding_fps"]

        if planning_fps <= 0:
            return float("inf")

        return total_frames / planning_fps

    def list_benchmarks(self):
        return [
            {
                "file_path": file_path,
                "codec": codec,
                "preset": preset,
                "crf": crf,
                "average_encoding_fps": benchmark["benchmark_summary"]["average_encoding_fps"],
                "stddev_encoding_fps": benchmark["benchmark_summary"]["stddev_encoding_fps"],
                "total_runs": benchmark["benchmark_summary"]["total_runs"],
            }
            for (file_path, codec, preset, crf), benchmark in sorted(self.benchmark.items(), key = lambda item: tuple(str(x) for x in item[0]))
        ]

    def list_merged_benchmarks(self):
        settings = sorted({key[1:] for key in self.benchmark})
        result = []

        for codec, preset, crf in settings:
            benchmark = self.get_benchmark(codec, preset, crf)
            summary = benchmark["benchmark_summary"]

            result.append({
                "codec": codec,
                "preset": preset,
                "crf": crf,
                "file_paths": benchmark["setting"]["file_paths"],
                "total_files": summary["total_files"],
                "average_encoding_fps": summary["average_encoding_fps"],
                "stddev_encoding_fps": summary["stddev_encoding_fps"],
                "total_runs": summary["total_runs"],
            })

        return result

class WorkerConflictError(ValueError):
    def __init__(self, duplicates, existing):
        self.duplicates = duplicates
        self.existing = existing
        parts = []

        if duplicates:
            parts.append(f"duplicate IDs in request: {duplicates}")

        if existing:
            parts.append(f"IDs already registered: {existing}")

        super().__init__("; ".join(parts))

class WorkerRegistry:
    def __init__(self):
        self.workers = {}
        self.lock = threading.RLock()

    def add_worker(self, id, address, port):
        with self.lock:
            if id in self.workers:
                raise ValueError(f"Worker with ID {id} already exists.")

            worker = WorkerProfile(id, address, port)
            self.workers[id] = worker
            return worker

    def add_workers(self, items):
        ids = [item["id"] for item in items]
        duplicates = sorted(i for i, n in Counter(ids).items() if n > 1)

        with self.lock:
            existing = sorted({i for i in ids if i in self.workers})

            if duplicates or existing:
                raise WorkerConflictError(duplicates, existing)

            added = []

            for item in items:
                worker = WorkerProfile(item["id"], item["address"], item["port"])
                self.workers[worker.id] = worker
                added.append(worker)

            return added

    def remove_worker(self, id):
        with self.lock:
            if id not in self.workers:
                raise ValueError(f"Worker with ID {id} does not exist.")

            return self.workers.pop(id)

    def get_worker(self, id):
        with self.lock:
            if id not in self.workers:
                raise ValueError(f"Worker with ID {id} does not exist.")

            return self.workers[id]

    def list_workers(self):
        with self.lock:
            return list(self.workers.values())

    def benchmark_worker_dummy(self, id, codec = "libx264", preset = "fast", crf = 24, avg_fps = 30.0, stddev_fps = 2.0, total_runs = 3, file_path = None, print_info = False):
        worker = self.get_worker(id)
        dummy_benchmark = {
            "setting": {
                "codec": codec,
                "preset": preset,
                "crf": crf,
                "file_path": file_path,
            },
            "benchmark_summary": {
                "average_encoding_fps": avg_fps,
                "stddev_encoding_fps": stddev_fps,
                "total_runs": total_runs,
            }
        }

        if print_info:
            print(f"\nDummy Benchmarking completed for file: {file_path}, codec: {codec}, preset: {preset}, crf: {crf}")
            print(f"Average encoding FPS: {avg_fps} fps")
            print(f"Encoding FPS stddev: {stddev_fps} fps")

        worker.add_benchmark(dummy_benchmark)
        return dummy_benchmark