import os
import time
import json
import uuid
import subprocess
import statistics

from dotenv import load_dotenv
from fractions import Fraction
from datetime import datetime

load_dotenv()

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/")
SHARE_DIR = os.environ.get("SHARE_DIR", ".")

def resolve_share_path(relative_path, share_dir = SHARE_DIR):
    root = os.path.realpath(share_dir)
    full_path = os.path.realpath(os.path.join(root, relative_path))

    if os.path.normcase(os.path.commonpath([root, full_path])) != os.path.normcase(root):
        raise ValueError(f"Path {relative_path} is outside the shared directory.")

    return full_path

class VideoInformation:
    def __init__(self, input_path, output_dir = OUTPUT_DIR, ffmpeg_path = FFMPEG_PATH, ffprobe_path = FFPROBE_PATH, print_info = False):
        self.input_path = input_path
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.print_info = print_info

    def run_command(self, command, tool, return_stderr = False):
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{tool} command failed with error: {e.stderr.strip()}") from e

        return result.stderr.strip() if return_stderr else result.stdout.strip()

    def color_space(self):
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=color_space,color_transfer,color_primaries",
            "-of", "json",
            self.input_path
        ]

        info = json.loads(self.run_command(command, "ffprobe"))
        stream = info.get("streams", [{}])[0]

        space = stream.get("color_space", "unknown")
        transfer = stream.get("color_transfer", "unknown")
        primaries = stream.get("color_primaries", "unknown")

        dynamics_range = "HDR" if transfer in ["smpte2084", "arib-std-b67", "smpte428", "hlg"] else "SDR"

        return {
            "color_space": space,
            "color_transfer": transfer,
            "color_primaries": primaries,
            "dynamics_range": dynamics_range
        }

    def video_info(self):
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,pix_fmt,bits_per_raw_sample,nb_frames",
            "-of", "json",
            self.input_path
        ]

        info = json.loads(self.run_command(command, "ffprobe"))

        stream = info.get("streams", [{}])[0]
        codec = stream.get("codec_name", "unknown")
        width = stream.get("width", 0)
        height = stream.get("height", 0)
        total_frames = int(stream.get("nb_frames", 0))

        frame_rate = stream.get("r_frame_rate", "0/1").split('/')
        frame_rate = float(frame_rate[0]) / float(frame_rate[1]) if len(frame_rate) == 2 and float(frame_rate[1]) != 0 else float(frame_rate[0])

        bit_depth = stream.get("bits_per_raw_sample", 0)
        if bit_depth:
            bit_depth = int(bit_depth)
        else:
            pix_fmt = stream.get("pix_fmt", "")
            if "yuv420p12le" in pix_fmt:
                bit_depth = 12
            elif "yuv420p10le" in pix_fmt:
                bit_depth = 10
            elif "yuv420p" in pix_fmt:
                bit_depth = 8
            else:
                bit_depth = 8

        return {
            "codec": codec,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
            "bit_depth": bit_depth,
            "total_frames": total_frames
        }

    def segment_splits(self):
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=time_base:packet=pts,flags",
            "-of", "json",
            self.input_path,
        ]
 
        info = json.loads(self.run_command(command, "ffprobe"))
 
        numerator, denominator = info["streams"][0]["time_base"].split("/")
        time_base = Fraction(int(numerator), int(denominator))
 
        packets = info.get("packets", [])
 
        if not packets:
            raise RuntimeError("No packets found in video")
 
        if any(packet.get("pts") is None for packet in packets):
            raise RuntimeError("Video has packets without pts, cannot split reliably")
 
        pts_values = [int(packet["pts"]) for packet in packets]
        keyframes = [i for i, packet in enumerate(packets) if "K" in packet.get("flags", "")]
 
        if not keyframes or keyframes[0] != 0:
            raise RuntimeError("Video does not start with a keyframe")
 
        starts = [0]
 
        for n, i in enumerate(keyframes[1:], start = 1):
            next_i = keyframes[n + 1] if n + 1 < len(keyframes) else len(packets)
 
            if all(pts >= pts_values[i] for pts in pts_values[i + 1:next_i]):
                starts.append(i)
 
        if len(starts) == 1 and len(keyframes) > 1:
            raise RuntimeError("No closed GOP keyframes found, re-encode the source with closed GOP (x265: open-gop=0)")
 
        sorted_pts = sorted(pts_values)
 
        if len(sorted_pts) > 1:
            frame_duration = min(b - a for a, b in zip(sorted_pts, sorted_pts[1:]))
        else:
            frame_duration = 0
 
        video_end = sorted_pts[-1] + frame_duration
 
        bounds = starts + [len(packets)]
        block_starts = [min(pts_values[bounds[n]:bounds[n + 1]]) for n in range(len(starts))]
 
        segments = []
 
        for n, first_frame in enumerate(starts):
            start = block_starts[n]
            end = block_starts[n + 1] if n + 1 < len(starts) else video_end
 
            segments.append({
                "segment": n + 1,
                "start": float(start * time_base),
                "end": float(end * time_base),
                "duration": float((end - start) * time_base),
                "start_frame": first_frame,
                "total_frames": bounds[n + 1] - first_frame,
            })
 
        return segments

    def get_video_information(self):
        info_data = self.video_info()
        color_space_data = self.color_space()
        segment_splits_data = self.segment_splits()

        if self.print_info:
            print("Video Color Space Information:")
            print(f"Color Space: {color_space_data['color_space']}")
            print(f"Color Transfer: {color_space_data['color_transfer']}")
            print(f"Color Primaries: {color_space_data['color_primaries']}")
            print(f"Dynamics Range: {color_space_data['dynamics_range']}")

            print("\nVideo Information:")
            print(f"Codec: {info_data['codec']}")
            print(f"Resolution: {info_data['width']}x{info_data['height']}")
            print(f"Frame Rate: {info_data['frame_rate']} fps")
            print(f"Bit Depth: {info_data['bit_depth']} bits")
            print(f"Total Frames: {info_data['total_frames']} frames")

            print("\nVideo Segments Information:")
            for segment in segment_splits_data:
                print(f"Segment {segment['segment']}: Start = {segment['start']}s, End = {segment['end']}s, Duration = {segment['duration']}s, Start Frame = {segment['start_frame']}, Total Frames = {segment['total_frames']}")

        return {
            "video_info": info_data,
            "color_space": color_space_data,
            "segment_splits": segment_splits_data
        }

class Benchmark:
    def __init__(self, video_path, output_dir = OUTPUT_DIR, ffmpeg_path = FFMPEG_PATH, ffprobe_path = FFPROBE_PATH, print_info = False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.print_info = print_info
        self.video_info = VideoInformation(video_path, output_dir, ffmpeg_path, ffprobe_path, print_info = print_info).get_video_information()

    def run_command(self, command, tool):
        try:
            subprocess.run(
                command,
                check = True,
                capture_output = True,
                text = True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{tool} failed with error: {e.stderr.strip()}") from e

    def benchmark_encoding(self, codec = "libx264", preset = "medium", crf = 23, repeat = 1, delay = 1):
        if repeat < 1:
            raise ValueError("repeat must be at least 1.")

        if delay < 0:
            raise ValueError("delay must not be negative.")

        results = []

        try:
            for i in range(repeat):
                command = [
                    self.ffmpeg_path,
                    "-hide_banner", "-nostats",
                    "-i", self.video_path,
                    "-c:v", codec,
                    "-preset", preset,
                    "-crf", str(crf),
                    "-f", "null",
                    "-"
                ]

                start_time = time.time()
                self.run_command(command, "ffmpeg")
                end_time = time.time()

                encoding_time = end_time - start_time
                results.append({
                    "run": i + 1,
                    "encoding_time": encoding_time,
                    "encoding_fps": self.video_info["video_info"]["total_frames"] / encoding_time if encoding_time > 0 else float('inf'),
                })

                if i < repeat - 1:
                    time.sleep(delay)
        except Exception as e:
            raise RuntimeError(f"Benchmarking failed: {str(e)}") from e

        if self.print_info:
            print(f"\nBenchmarking completed for codec: {codec}, preset: {preset}, crf: {crf}")
            print(f"Total runs: {len(results)}")

            print(f"Min encoding time: {min([r['encoding_time'] for r in results]) if results else 0} seconds")
            print(f"Max encoding time: {max([r['encoding_time'] for r in results]) if results else 0} seconds")
            print(f"Average encoding time: {statistics.mean([r['encoding_time'] for r in results]) if results else 0} seconds")
            print(f"Encoding time stddev: {statistics.stdev([r['encoding_time'] for r in results]) if len([r['encoding_time'] for r in results]) > 1 else 0} seconds")
            
            print(f"Min encoding FPS: {min([r['encoding_fps'] for r in results]) if results else 0} fps")
            print(f"Max encoding FPS: {max([r['encoding_fps'] for r in results]) if results else 0} fps")
            print(f"Average encoding FPS: {statistics.mean([r['encoding_fps'] for r in results]) if results else 0} fps")
            print(f"Encoding FPS stddev: {statistics.stdev([r['encoding_fps'] for r in results]) if len([r['encoding_fps'] for r in results]) > 1 else 0} fps")

            print("\nIndividual Run Results:")

            for result in results:
                print(f"Run {result['run']}: Encoding Time = {result['encoding_time']} seconds, Encoding FPS = {result['encoding_fps']} fps")

        return {
            "video_info": self.video_info,
            "setting": {
                "codec": codec,
                "preset": preset,
                "crf": crf,
            },
            "results": results,
            "benchmark_summary": {
                "total_runs": len(results),
                "average_encoding_time": statistics.mean([r["encoding_time"] for r in results]) if results else 0,
                "stddev_encoding_time": statistics.stdev([r["encoding_time"] for r in results]) if len([r["encoding_time"] for r in results]) > 1 else 0,
                "min_encoding_time": min([r["encoding_time"] for r in results]) if results else 0,
                "max_encoding_time": max([r["encoding_time"] for r in results]) if results else 0,
                "average_encoding_fps": statistics.mean([r["encoding_fps"] for r in results]) if results else 0,
                "stddev_encoding_fps": statistics.stdev([r["encoding_fps"] for r in results]) if len([r["encoding_fps"] for r in results]) > 1 else 0,
                "min_encoding_fps": min([r["encoding_fps"] for r in results]) if results else 0,
                "max_encoding_fps": max([r["encoding_fps"] for r in results]) if results else 0,
            }
        }

class SegmentEncoding:
    def __init__(self, video_path, output_dir = OUTPUT_DIR, ffmpeg_path = FFMPEG_PATH, ffprobe_path = FFPROBE_PATH, print_info = False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.print_info = print_info

    def run_command(self, command, tool):
        try:
            result = subprocess.run(
                command,
                check = True,
                capture_output = True,
                text = True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{tool} failed with error: {e.stderr.strip()}") from e

        return result.stdout.strip()

    def get_output_path(self, segment, extension = "mp4"):
        return os.path.join(self.output_dir, f"segment_{segment['segment']:04d}.{extension}")

    def build_command(self, segment, output_path, codec, preset, crf):
        # segment dipotong langsung dari video asli (tanpa file potongan) dengan -ss di keyframe closed GOP.
        # Seek dimundurkan setengah frame supaya frame keyframe tidak terbuang oleh pembulatan -ss,
        # lalu jumlah frame dikunci dengan -frames:v supaya hasilnya tepat total_frames.
        frame_duration = segment["duration"] / segment["total_frames"]
        seek = max(0.0, segment["start"] - frame_duration / 2)

        return [
            self.ffmpeg_path,
            "-hide_banner", "-nostats", "-loglevel", "error", "-y",
            "-ss", f"{seek:.6f}",
            "-i", self.video_path,
            "-map", "0:v:0",
            "-frames:v", str(segment["total_frames"]),
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            output_path,
        ]

    def count_frames(self, path):
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "json",
            path,
        ]

        info = json.loads(self.run_command(command, "ffprobe"))
        return int(info["streams"][0]["nb_read_packets"])

    def encode_segment(self, segment, codec = "libx264", preset = "fast", crf = 24, verify = True):
        os.makedirs(self.output_dir, exist_ok = True)

        output_path = self.get_output_path(segment)
        temp_path = f"{output_path}.{uuid.uuid4().hex[:8]}.part.mp4"
        command = self.build_command(segment, temp_path, codec, preset, crf)

        try:
            start_time = time.perf_counter()
            self.run_command(command, "ffmpeg")
            encoding_time = time.perf_counter() - start_time

            if verify:
                frames = self.count_frames(temp_path)

                if frames != segment["total_frames"]:
                    raise RuntimeError(f"Segment {segment['segment']} has {frames} frames, expected {segment['total_frames']}.")

            # hasil baru dipublikasikan setelah lolos verifikasi, jadi file setengah jadi tidak pernah terlihat master
            os.replace(temp_path, output_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        encoding_fps = segment["total_frames"] / encoding_time

        if self.print_info:
            print(f"Segment {segment['segment']} encoded: {segment['total_frames']} frames in {encoding_time} seconds ({encoding_fps} fps) -> {output_path}")

        return {
            "segment": segment["segment"],
            "total_frames": segment["total_frames"],
            "encoding_time": encoding_time,
            "encoding_fps": encoding_fps,
            "output_file": os.path.basename(output_path),
            "output_path": output_path,
        }

class CombineSegments:
    def __init__(self, output_dir = OUTPUT_DIR, ffmpeg_path = FFMPEG_PATH, print_info = False):
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path
        self.print_info = print_info

    def run_command(self, command, tool):
        try:
            subprocess.run(
                command,
                check = True,
                capture_output = True,
                text = True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{tool} failed with error: {e.stderr.strip()}") from e

    def combine(self, files, video_path, job_id, delete_segments = True):
        missing = [f for f in files if not os.path.isfile(f)]

        if missing:
            raise RuntimeError(f"File segmen tidak ditemukan: {missing[:3]}")

        os.makedirs(self.output_dir, exist_ok = True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        extension = os.path.splitext(files[0])[1] or ".mp4"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"{base_name}_{timestamp}{extension}")
        list_path = os.path.join(self.output_dir, f"concat_{job_id}.txt")

        with open(list_path, "w", encoding = "utf-8") as f:
            for path in files:
                escaped = os.path.abspath(path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        command = [
            self.ffmpeg_path,
            "-hide_banner", "-nostats", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ]

        try:
            self.run_command(command, "ffmpeg")
        except RuntimeError:
            if os.path.isfile(output_path):
                os.remove(output_path)

            raise
        finally:
            os.remove(list_path)

        if delete_segments:
            for path in files:
                try:
                    os.remove(path)
                except OSError:
                    pass

        if self.print_info:
            print(f"Combined {len(files)} segments -> {output_path}")

        return output_path