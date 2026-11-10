from collections import deque

class BaseDistributor:
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

            if avg_fps <= 0:
                raise ValueError(f"Worker {worker.id} has non-positive FPS ({avg_fps}) for codec: {codec}, preset: {preset}, crf: {crf}")

            self.state[worker.id] = {
                "active": True,
                "avg_fps": avg_fps,
                "stddev_fps": stddev_fps,
                "sample_count": worker.get_sample_count(codec, preset, crf),
                "planning_fps": avg_fps,
                "remaining_frames": 0,
                "assigned_segments": [],
                "completed_segments": [],
            }

        if not self.state:
            raise ValueError("No valid workers available for distribution.")

        self.plan = {worker_id: deque(segments) for worker_id, segments in self.build_plan().items()}

    def build_plan(self):
        raise NotImplementedError

    def orphan_choice(self, orphans):
        return orphans[0]

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

    def assign_to_worker(self, worker_id, segment):
        if not self.state[worker_id]["active"]:
            raise ValueError(f"Worker {worker_id} is not active.")

        if segment not in self.pending_segments:
            raise ValueError(f"Segment {segment['segment']} is not pending.")

        self.add_load(worker_id, segment)
        self.pending_segments.remove(segment)

    def pop_planned(self, worker_id):
        queue = self.plan[worker_id]

        while queue:
            segment = queue.popleft()

            if segment in self.pending_segments:
                return segment

        return None

    def pick_orphan(self):
        planned = {s["segment"] for queue in self.plan.values() for s in queue}
        orphans = [s for s in self.pending_segments if s["segment"] not in planned]

        if not orphans:
            return None

        return self.orphan_choice(orphans)

    def initial_assignment(self, initial_size = 1):
        if initial_size < 1:
            raise ValueError("initial_size must be at least 1.")

        result = {worker_id: [] for worker_id in self.state}

        for worker_id in self.get_active_workers():
            for _ in range(initial_size):
                segment = self.pop_planned(worker_id)

                if segment is None:
                    break

                self.assign_to_worker(worker_id, segment)
                result[worker_id].append(segment)

        return result

    def assign_next(self, worker_id):
        if not self.state[worker_id]["active"] or not self.pending_segments:
            return None

        segment = self.pop_planned(worker_id)

        if segment is None:
            segment = self.pick_orphan()

        if segment is None:
            return None

        self.assign_to_worker(worker_id, segment)
        return segment

    def complete_segment(self, worker_id, segment, encoding_time):
        if encoding_time <= 0:
            raise ValueError("Encoding time must be greater than zero.")

        self.release_load(worker_id, segment)

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

        self.plan[worker_id].clear()

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

class FIFODistributor(BaseDistributor):
    def build_plan(self):
        worker_ids = list(self.state)
        plan = {worker_id: [] for worker_id in worker_ids}

        for index, segment in enumerate(self.segments):
            plan[worker_ids[index % len(worker_ids)]].append(segment)

        return plan

class SJFDistributor(BaseDistributor):
    def build_plan(self):
        worker_ids = list(self.state)
        plan = {worker_id: [] for worker_id in worker_ids}

        for index, segment in enumerate(sorted(self.segments, key = lambda s: s["total_frames"])):
            plan[worker_ids[index % len(worker_ids)]].append(segment)

        return plan

    def orphan_choice(self, orphans):
        return min(orphans, key = lambda s: s["total_frames"])

class HeterogeneousDistributor(BaseDistributor):
    def build_plan(self):
        worker_ids = list(self.state)
        plan = {worker_id: [] for worker_id in worker_ids}
        load = {worker_id: 0.0 for worker_id in worker_ids}

        for segment in sorted(self.segments, key = lambda s: s["total_frames"], reverse = True):
            best_worker = None
            best_finish_time = float("inf")

            for worker_id in worker_ids:
                finish_time = load[worker_id] + segment["total_frames"] / self.state[worker_id]["planning_fps"]

                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_worker = worker_id

            plan[best_worker].append(segment)
            load[best_worker] = best_finish_time

        self.estimated_load = load
        self.estimated_makespan = max(load.values())

        return plan

    def orphan_choice(self, orphans):
        return max(orphans, key = lambda s: s["total_frames"])