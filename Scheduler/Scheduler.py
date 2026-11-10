import math

class AdaptiveDistributor:
    def __init__(self, workers, video_information, codec = "libx264", preset = "fast", crf = 24, max_retries = 3):
        self.workers = workers
        self.segments = video_information["segment_splits"]
        self.video_info = video_information["video_info"]
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.max_retries = max_retries
        self.pending_segments = list(self.segments)
        self.failed_segments = []
        self.retry_count = {}
        self.state = {}

        for worker in workers.list_workers():
            if not worker.has_benchmark(codec, preset, crf):
                raise ValueError(f"Worker {worker.id} does not have benchmark data for codec: {codec}, preset: {preset}, crf: {crf}")

            avg_fps = worker.get_fps(codec, preset, crf)
            stddev_fps = worker.get_fps_stddev(codec, preset, crf)
            planning_fps = avg_fps - stddev_fps

            if planning_fps <= 0:
                raise ValueError(f"Worker {worker.id} has non-positive planning FPS ({planning_fps}) for codec: {codec}, preset: {preset}, crf: {crf}")

            self.state[worker.id] = {
                "active": True,
                "avg_fps": avg_fps,
                "stddev_fps": stddev_fps,
                "sample_count": worker.get_sample_count(codec, preset, crf),
                "planning_fps": planning_fps,
                "remaining_frames": 0,
                "assigned_segments": [],
                "completed_segments": [],
            }

        if not self.state:
            raise ValueError("No valid workers available for distribution.")

    def get_estimated_time(self, worker_id):
        if worker_id not in self.state:
            raise ValueError(f"Worker {worker_id} is not part of the distribution state.")

        worker_state = self.state[worker_id]
        return worker_state["remaining_frames"] / worker_state["planning_fps"]

    def check_running(self, worker_id, segment):
        worker_state = self.state[worker_id]

        if segment not in worker_state["assigned_segments"]:
            raise ValueError(f"Segment {segment['segment']} was not assigned to worker {worker_id}.")

        if segment in worker_state["completed_segments"]:
            raise ValueError(f"Segment {segment['segment']} is already completed on worker {worker_id}.")

    def add_load(self, worker_id, segment):
        if not segment:
            raise ValueError("No segment provided for assignment.")

        worker_state = self.state[worker_id]
        worker_state["remaining_frames"] += segment["total_frames"]
        worker_state["assigned_segments"].append(segment)

    def release_load(self, worker_id, segment):
        if not segment:
            raise ValueError("No segment provided for release.")

        self.check_running(worker_id, segment)
        worker_state = self.state[worker_id]
        worker_state["remaining_frames"] -= segment["total_frames"]
        worker_state["completed_segments"].append(segment)

    def get_running_segments(self, worker_id):
        worker_state = self.state[worker_id]
        return [s for s in worker_state["assigned_segments"] if s not in worker_state["completed_segments"]]

    def get_active_workers(self):
        return [worker_id for worker_id, worker_state in self.state.items() if worker_state["active"]]

    def get_coefficient(self, worker_id):
        worker_state = self.state[worker_id]
        return worker_state["stddev_fps"] / worker_state["avg_fps"]

    def get_finish_time(self, worker_id, segment):
        worker_state = self.state[worker_id]
        return (worker_state["remaining_frames"] + segment["total_frames"]) / worker_state["planning_fps"]

    def select_worker(self, segment):
        active_workers = self.get_active_workers()

        if not active_workers:
            raise RuntimeError("No active worker available.")

        return min(
            active_workers,
            key = lambda worker_id: (
                self.get_finish_time(worker_id, segment),
                self.get_coefficient(worker_id),
                worker_id,
            ),
        )

    def assign_to_worker(self, worker_id, segment):
        if not self.state[worker_id]["active"]:
            raise ValueError(f"Worker {worker_id} is not active.")

        if segment not in self.pending_segments:
            raise ValueError(f"Segment {segment['segment']} is not pending.")

        self.add_load(worker_id, segment)
        self.pending_segments.remove(segment)

    def assign_segment(self, segment):
        worker_id = self.select_worker(segment)
        self.assign_to_worker(worker_id, segment)
        return worker_id

    def initial_assignment(self, initial_size = 1):
        if initial_size < 1:
            raise ValueError("initial_size must be at least 1.")

        ordered_workers = sorted(
            self.get_active_workers(),
            key = lambda worker_id: self.state[worker_id]["planning_fps"],
            reverse = True,
        )
        result = {worker_id: [] for worker_id in self.state}

        for _ in range(initial_size):
            for worker_id in ordered_workers:
                if not self.pending_segments:
                    return result

                segment = max(self.pending_segments, key = lambda s: s["total_frames"])
                self.assign_to_worker(worker_id, segment)
                result[worker_id].append(segment)

        return result

    def get_next_segment(self, worker_id):
        if not self.state[worker_id]["active"] or not self.pending_segments:
            return None

        candidates = sorted(self.pending_segments, key = lambda s: s["total_frames"], reverse = True)

        for segment in candidates:
            if self.select_worker(segment) == worker_id:
                return segment

        return candidates[-1]

    def assign_next(self, worker_id):
        segment = self.get_next_segment(worker_id)

        if segment is None:
            return None

        self.assign_to_worker(worker_id, segment)
        return segment

    def update_fps(self, worker_id, measured_fps):
        worker_state = self.state[worker_id]
        n = worker_state["sample_count"]
        mean = worker_state["avg_fps"]
        m2 = (worker_state["stddev_fps"] ** 2) * (n - 1)

        n += 1
        delta = measured_fps - mean
        mean += delta / n
        m2 += delta * (measured_fps - mean)

        worker_state["sample_count"] = n
        worker_state["avg_fps"] = mean
        worker_state["stddev_fps"] = math.sqrt(m2 / (n - 1))
        worker_state["planning_fps"] = max(1e-6, mean - worker_state["stddev_fps"])

    def complete_segment(self, worker_id, segment, encoding_time):
        if encoding_time <= 0:
            raise ValueError("Encoding time must be greater than zero.")

        self.release_load(worker_id, segment)
        self.update_fps(worker_id, segment["total_frames"] / encoding_time)

    def fail_segment(self, worker_id, segment, count_retry = True):
        self.check_running(worker_id, segment)

        worker_state = self.state[worker_id]
        worker_state["assigned_segments"].remove(segment)
        worker_state["remaining_frames"] -= segment["total_frames"]

        if count_retry:
            segment_id = segment["segment"]
            self.retry_count[segment_id] = self.retry_count.get(segment_id, 0) + 1

            if self.retry_count[segment_id] > self.max_retries:
                self.failed_segments.append(segment)
                return False

        self.pending_segments.append(segment)
        return True

    def disable_worker(self, worker_id):
        self.state[worker_id]["active"] = False
        returned_segments = []

        for segment in self.get_running_segments(worker_id):
            if self.fail_segment(worker_id, segment, count_retry = False):
                returned_segments.append(segment)

        return returned_segments

    def get_idle_workers(self):
        return [worker_id for worker_id in self.get_active_workers() if not self.get_running_segments(worker_id)]

    def is_finished(self):
        if self.pending_segments:
            return False

        return all(not self.get_running_segments(worker_id) for worker_id in self.state)

    def get_status(self):
        completed = sum(len(s["completed_segments"]) for s in self.state.values())
        running = sum(len(self.get_running_segments(worker_id)) for worker_id in self.state)

        return {
            "total_segments": len(self.segments),
            "completed": completed,
            "running": running,
            "pending": len(self.pending_segments),
            "failed": len(self.failed_segments),
            "finished": self.is_finished(),
            "workers": {
                worker_id: {
                    "active": s["active"],
                    "avg_fps": s["avg_fps"],
                    "stddev_fps": s["stddev_fps"],
                    "cv": self.get_coefficient(worker_id),
                    "planning_fps": s["planning_fps"],
                    "sample_count": s["sample_count"],
                    "remaining_frames": s["remaining_frames"],
                    "estimated_time": self.get_estimated_time(worker_id),
                    "running_segments": [seg["segment"] for seg in self.get_running_segments(worker_id)],
                    "completed_segments": [seg["segment"] for seg in s["completed_segments"]],
                }
                for worker_id, s in self.state.items()
            },
        }