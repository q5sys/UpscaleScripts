#!/usr/bin/env python3
"""
SeedVR2 phase-split multi-GPU production runner.

Pipeline
========

Stage A -- pre-encode:
    GPU 3  RTX PRO 6000
    GPU 0  RTX 5090
    GPU 1  RTX 5090
    GPU 4  RTX 5070 Ti

    All four GPUs VAE-encode independent 65-frame sliding windows using the
    same 512/64 tiling configuration. Encoded latents are cached to a RAM-backed
    directory.

Stage B+C -- concurrent restore/decode:
    GPU 3  RTX PRO 6000 only: FP16 7B DiT producer
    GPU 0  RTX 5090:          VAE decode consumer
    GPU 1  RTX 5090:          VAE decode consumer
    GPU 4  RTX 5070 Ti:       VAE decode consumer

    As soon as the 6000 atomically writes a restored latent, that window becomes
    eligible for decode. DiT and decode therefore run concurrently. Existing
    restored latents from an interrupted run are consumed immediately while the
    6000 resumes only missing DiT windows.

    Completed decoded windows are buffered in a bounded RAM-disk queue,
    assembled in source order with the same raised-cosine overlap blend used by
    seedvr2_sliding_window_v13.py, streamed to HEVC NVENC, then original
    audio/subtitle/data/attachment streams and chapters/metadata are remuxed.

Important design choices
========================
* Window size is global for the whole run: default 65 frames, overlap 4.
* Every encode worker uses identical VAE tiling (512/64).
* Every decode worker uses identical VAE tiling (512/64).
* Each sliding window is treated as an atomic VAE job. Overlapping source frames
  are intentionally encoded independently, matching the existing v13 pipeline.
* SeedVR2's existing deterministic seed behavior is retained: every atomic
  window receives the same base seed. The backend itself uses seed+1,000,000
  for VAE sampling and resets the DiT seed for each batch.
* The 5070 Ti is opportunistic. If it OOMs during decode, its failed job is
  automatically requeued and the remaining two 5090 workers continue.
* Encoded/restored latent caches are resumable. Stage B+C starts a fresh video
  assembly on each invocation but reuses all completed encoded/restored caches.
* The default NVENC device is GPU 0 so GPU 3 remains dedicated to DiT.
* Phase 4/LAB is memory-bounded into 17-frame groups after VAE decode;
  the 65-frame VAE/DiT temporal inference window remains unchanged.
* Every long-lived worker performs SeedVR2 deep allocator/GC cleanup after
  each atomic window while preserving the cached model runner.
* Source/cache locations are portable: source content is fingerprinted instead
  of absolute path/mtime and existing moved caches can be adopted in place.
* Failed H.264 per-window seeks automatically retry with decoder preroll,
  preserving already-cached windows and recovering difficult open-GOP/EOF seeks.

This is intentionally built against the NumZ backend private phase API and
seedvr2_sliding_window_v13.py compatibility patches. Pin the backend checkout
used for validation; rerun equivalence tests after backend changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional


SCRIPT_VERSION = "2026.08.13-phase-split-v28"
CACHE_FORMAT_VERSION = 1
SOURCE_IDENTITY_VERSION = 1
SOURCE_FINGERPRINT_CHUNK_BYTES = 4 * 1024 * 1024
SOURCE_FINGERPRINT_SAMPLES = 5

# Per-window source decode normally uses the existing direct accurate seek.
# If an H.264/open-GOP seek near EOF cannot produce frames, retry only that
# window from progressively earlier points and frame-trim to the target.
SOURCE_SEEK_PREROLL_RETRIES_SECONDS = (5.0, 30.0, 120.0)

# ---------------------------------------------------------------------------
# Default environment / hardware
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE = Path("/zstor/ai/seedvr2")
DEFAULT_PYTHON = Path("/zstor/conda/envs/seedvr2-nvfp4/bin/python")
DEFAULT_BACKEND_DIR = DEFAULT_WORKSPACE / "ComfyUI/custom_nodes/seedvr2_videoupscaler"
DEFAULT_COMFYUI_DIR = DEFAULT_WORKSPACE / "ComfyUI"
DEFAULT_V13_RUNNER = DEFAULT_WORKSPACE / "seedvr2_sliding_window_v13.py"

# ---------------------------------------------------------------------------
# USER CONFIGURATION
#
# Edit these directly if desired. --input, --output and --cache-dir remain
# available as command-line overrides.
# ---------------------------------------------------------------------------

INPUT_FILE = Path("/mnt/ramdisk/The.Last.Samurai.m2ts")
OUTPUT_FILE = Path("/mnt/ramdisk/The.Last.Samurai-SEEDVR2-AIU.mkv")
CACHE_DIR = Path("/mnt/ramdisk/lastsamurai-phase-cache")

DEFAULT_MODEL_DIR = Path("/zstor/ai/seedvr2/seedvr2/models/")
DEFAULT_DIT_MODEL = "seedvr2_ema_7b_fp16.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"

DEFAULT_ENCODE_GPUS = ("3", "0", "1", "4")
DEFAULT_DIT_GPU = "3"
DEFAULT_DECODE_GPUS = ("0", "1", "4")
DEFAULT_NVENC_GPU = "0"

DEFAULT_WINDOW = 65
DEFAULT_OVERLAP = 4
DEFAULT_RESOLUTION = 2160
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 64
DEFAULT_SEED = 42
DEFAULT_INPUT_NOISE_SCALE = 0.0
DEFAULT_LATENT_NOISE_SCALE = 0.0

SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm",
    ".m2ts", ".ts", ".mts", ".mpg", ".mpeg", ".wmv", ".flv", ".mxf",
)

# Decode buffering is bounded because a decoded 65-frame 4K FP16 window is
# roughly 3 GiB. Six completed/in-flight jobs keeps RAM-disk pressure modest.
DEFAULT_DECODE_INFLIGHT = 6

# LAB Phase 4 is much more memory intensive than VAE decode because it
# converts content/reference tensors to float32. Keep temporal model
# inference at 65f, but color-correct in bounded 17f groups.
DEFAULT_POSTPROCESS_CHUNK_FRAMES = 17

# Deep SeedVR2 cleanup runs after every atomic worker job. RSS is reported
# periodically so long-run host-memory retention is visible in the log.
WORKER_MEMORY_LOG_INTERVAL = 25

# Free-VRAM guards. These are deliberately conservative and can be overridden.
DEFAULT_ENCODE_MIN_FREE_MIB = {
    "3": 18000,  # 6000 encode with tiled VAE
    "0": 18000,
    "1": 18000,
    "4": 14500,
}
DEFAULT_DIT_MIN_FREE_MIB = 76000
DEFAULT_DECODE_MIN_FREE_MIB = {
    "0": 24000,
    "1": 24000,
    "4": 14500,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    frames: int
    duration: float

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def fps_text(self) -> str:
        return f"{self.fps_num}/{self.fps_den}"


@dataclass(frozen=True)
class WindowJob:
    window_id: int
    shot_id: int
    start_frame: int
    real_count: int
    new_count: int
    is_first: bool
    is_last: bool
    shot_frames: int

    @property
    def name(self) -> str:
        return f"window_{self.window_id:06d}"


@dataclass
class Config:
    input: str
    output: str
    cache_dir: str
    workspace: str
    backend_dir: str
    comfyui_dir: str
    v13_runner: str
    model_dir: str
    dit_model: str
    vae_model: str

    encode_gpus: list[str]
    dit_gpu: str
    decode_gpus: list[str]
    nvenc_gpu: str

    window_frames: int
    overlap_frames: int
    resolution: int
    max_resolution: int
    encode_tile_size: int
    encode_tile_overlap: int
    decode_tile_size: int
    decode_tile_overlap: int

    attention_mode: str
    color_correction: str
    input_noise_scale: float
    latent_noise_scale: float
    seed: int

    scene_detect: bool
    scene_threshold: float
    test_start: float
    test_seconds: float

    nvenc_preset: str
    nvenc_cq: int
    decode_inflight: int
    postprocess_chunk_frames: int

    keep_encoded: bool
    keep_restored: bool
    keep_decoded: bool
    fresh_cache: bool
    debug: bool

    dit_min_free_mib: int
    encode_min_free_mib: dict[str, int]
    decode_min_free_mib: dict[str, int]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def eprint(*args: object, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


def log(*args: object) -> None:
    print(*args, flush=True)


def die(message: str, code: int = 1) -> "NoReturn":
    eprint(f"ERROR: {message}")
    raise SystemExit(code)


def require_program(name_or_path: str) -> str:
    path = Path(name_or_path)
    if path.is_absolute() and os.access(path, os.X_OK):
        return str(path)
    found = shutil.which(name_or_path)
    if not found:
        die(f"Required executable not found: {name_or_path}")
    return found


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(block_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".part-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".part-{os.getpid()}")
    torch.save(payload, temp)
    os.replace(temp, path)


def torch_load_cpu(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_rate(value: str) -> Fraction:
    if "/" in value:
        num, den = value.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(value)


def probe_video(path: Path) -> VideoInfo:
    """
    Probe video geometry/rate and COUNT the frames FFprobe can actually read.

    Container-reported nb_frames is not reliable enough for constructing the
    final atomic source window. In particular, an overstated nb_frames can make
    the last planned window seek beyond EOF and return a zero-byte FFmpeg read.

    Prefer nb_read_frames from -count_frames. Fall back to nb_frames, then to
    duration*fps only when a counted value is unavailable.
    """
    ffprobe = require_program("/usr/sbin/ffprobe")
    log(
        "Counting readable source frames with FFprobe "
        "(one-time full-video scan for a new cache)..."
    )
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames:format=duration",
        "-of", "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    if not data.get("streams"):
        die(f"No video stream found: {path}")

    stream = data["streams"][0]
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    rate = parse_rate(rate_text)
    if rate <= 0:
        die(f"Invalid frame rate from ffprobe: {rate_text}")

    duration = float(data.get("format", {}).get("duration") or 0.0)

    reported_text = stream.get("nb_frames")
    counted_text = stream.get("nb_read_frames")

    reported_frames = (
        int(reported_text)
        if reported_text and reported_text not in {"N/A", "0"}
        else None
    )
    counted_frames = (
        int(counted_text)
        if counted_text and counted_text not in {"N/A", "0"}
        else None
    )

    if counted_frames is not None:
        frames = counted_frames
        if reported_frames is not None and reported_frames != counted_frames:
            eprint(
                "Frame-count correction: container reports "
                f"{reported_frames} frames, FFprobe counted {counted_frames} "
                "readable frames; using the counted value."
            )
        else:
            eprint(f"Verified readable video frames: {counted_frames}")
    elif reported_frames is not None:
        frames = reported_frames
        eprint(
            "WARNING: FFprobe did not return nb_read_frames; falling back to "
            f"container nb_frames={reported_frames}."
        )
    elif duration > 0:
        frames = int(round(duration * float(rate)))
        eprint(
            "WARNING: FFprobe returned neither counted nor reported frame "
            f"count; using duration*fps={frames}."
        )
    else:
        die("Could not determine source frame count")

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        frames=frames,
        duration=duration,
    )


def detect_scene_boundaries(path: Path, info: VideoInfo, threshold: float) -> list[int]:
    """Return hard-cut frame boundaries for the full source."""
    if threshold <= 0:
        return [0, info.frames]

    ffmpeg = require_program("/usr/sbin/ffmpeg")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "info",
        "-i", str(path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-an", "-sn", "-dn",
        "-f", "null", "-",
    ]
    eprint("Scanning source for hard scene cuts...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    assert proc.stderr is not None

    import re
    pts_pattern = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
    cuts = {0, info.frames}
    for line in proc.stderr:
        match = pts_pattern.search(line)
        if not match:
            continue
        frame = int(round(float(match.group(1)) * info.fps))
        if 1 <= frame < info.frames:
            cuts.add(frame)

    status = proc.wait()
    if status != 0:
        die(f"FFmpeg scene detection failed with exit status {status}")
    return sorted(cuts)


def valid_seedvr2_batch(value: int) -> bool:
    return value >= 5 and (value - 1) % 4 == 0


def gpu_query(gpu: str) -> tuple[str, int, int]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    parts = [x.strip() for x in out.split(",")]
    if len(parts) != 3:
        die(f"Unexpected nvidia-smi output for GPU {gpu}: {out}")
    return parts[0], int(parts[1]), int(parts[2])


def check_gpu(gpu: str, minimum_free_mib: int, expected: Optional[str] = None) -> None:
    name, total, free = gpu_query(gpu)
    if expected and expected not in name:
        eprint(f"WARNING: GPU {gpu} is {name!r}, expected name containing {expected!r}")
    if free < minimum_free_mib:
        die(
            f"GPU {gpu} ({name}) has only {free} MiB free; "
            f"{minimum_free_mib} MiB is required for this stage"
        )
    log(f"  GPU {gpu}: {name} | {free}/{total} MiB free")


def cache_disk_status(cache_dir: Path) -> str:
    usage = shutil.disk_usage(cache_dir)
    return (
        f"{usage.free / (1024**3):.1f} GiB free / "
        f"{usage.total / (1024**3):.1f} GiB total"
    )


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------

def make_shot_jobs(
    shot_id: int,
    shot_start: int,
    shot_end: int,
    *,
    window: int,
    overlap: int,
    first_window_id: int,
) -> list[WindowJob]:
    shot_frames = shot_end - shot_start
    if shot_frames <= 0:
        return []

    hop = window - overlap
    jobs: list[WindowJob] = []

    if shot_frames <= window:
        jobs.append(
            WindowJob(
                window_id=first_window_id,
                shot_id=shot_id,
                start_frame=shot_start,
                real_count=shot_frames,
                new_count=shot_frames,
                is_first=True,
                is_last=True,
                shot_frames=shot_frames,
            )
        )
        return jobs

    # First full window.
    jobs.append(
        WindowJob(
            window_id=first_window_id,
            shot_id=shot_id,
            start_frame=shot_start,
            real_count=window,
            new_count=hop,
            is_first=True,
            is_last=False,
            shot_frames=shot_frames,
        )
    )

    remaining_new = shot_frames - window
    current_start = shot_start + hop
    next_id = first_window_id + 1

    while remaining_new > 0:
        new_count = min(hop, remaining_new)
        real_count = overlap + new_count
        is_last = remaining_new <= hop
        jobs.append(
            WindowJob(
                window_id=next_id,
                shot_id=shot_id,
                start_frame=current_start,
                real_count=real_count,
                new_count=new_count,
                is_first=False,
                is_last=is_last,
                shot_frames=shot_frames,
            )
        )
        remaining_new -= new_count
        current_start += hop
        next_id += 1

    return jobs


def build_jobs(
    info: VideoInfo,
    *,
    boundaries: list[int],
    window: int,
    overlap: int,
    test_start: float,
    test_seconds: float,
) -> list[WindowJob]:
    if test_seconds > 0:
        start_frame = max(0, int(round(test_start * info.fps)))
        end_frame = min(info.frames, start_frame + int(round(test_seconds * info.fps)))
        boundaries = [start_frame, end_frame]

    jobs: list[WindowJob] = []
    next_id = 0
    for shot_id, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        shot_jobs = make_shot_jobs(
            shot_id,
            start,
            end,
            window=window,
            overlap=overlap,
            first_window_id=next_id,
        )
        jobs.extend(shot_jobs)
        next_id += len(shot_jobs)
    return jobs


def expected_output_frames(jobs: list[WindowJob], overlap: int) -> int:
    total = 0
    by_shot: dict[int, list[WindowJob]] = {}
    for job in jobs:
        by_shot.setdefault(job.shot_id, []).append(job)

    for shot_jobs in by_shot.values():
        if len(shot_jobs) == 1:
            total += shot_jobs[0].real_count
            continue

        # The first window emits hop frames and holds its final overlap.
        total += shot_jobs[0].new_count

        # Every middle window resolves the previous overlap and emits enough
        # additional frames to advance exactly one hop.
        for job in shot_jobs[1:-1]:
            total += job.new_count

        # The final window resolves the pending overlap and emits its final new
        # frames, i.e. its entire real_count.
        total += shot_jobs[-1].real_count

    return total


# ---------------------------------------------------------------------------
# Backend bootstrap -- worker processes only
# ---------------------------------------------------------------------------

@dataclass
class BackendBundle:
    torch: Any
    np: Any
    seed_cli: Any
    debug: Any
    setup_generation_context: Any
    prepare_runner: Any
    compute_generation_info: Any
    encode_all_batches: Any
    upscale_all_batches: Any
    decode_all_batches: Any
    postprocess_all_batches: Any
    manage_model_device: Any
    clear_memory: Any
    get_ram_usage: Any
    runner: Any
    cache_context: dict[str, Any]


def _load_v13_module(path: Path) -> Any:
    """
    Load the v13 compatibility runner as a real registered Python module.

    Python 3.13 dataclasses resolves annotation types through
    sys.modules[cls.__module__] while class decorators execute. A module created
    with module_from_spec() must therefore be inserted into sys.modules before
    exec_module(); otherwise @dataclass can fail with:

        AttributeError: 'NoneType' object has no attribute '__dict__'
    """
    module_name = "seedvr2_sliding_window_v13_compat"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v13 compatibility runner from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Do not leave a partially initialized module behind if import fails.
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module


def _patch_namespace(cfg: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        dit_model=cfg["dit_model"],
        vae_model=cfg["vae_model"],
        model_dir=Path(cfg["model_dir"]),
        model_size="7b",
        nvfp4_loader="none",
        attention_mode=cfg["attention_mode"],
        comfyui_dir=Path(cfg["comfyui_dir"]),
        allow_model_download=False,
        debug=cfg["debug"],
    )


def bootstrap_backend(cfg: dict[str, Any], physical_gpu: str, phase: str) -> BackendBundle:
    # Critical: do this before importing torch, ComfyUI or SeedVR2.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ["SEEDVR2_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    backend_dir = Path(cfg["backend_dir"])
    comfyui_dir = Path(cfg["comfyui_dir"])
    v13_path = Path(cfg["v13_runner"])

    v13 = _load_v13_module(v13_path)
    seed_cli = v13.import_seedvr2(
        backend_dir,
        cfg["attention_mode"],
        comfyui_dir,
    )
    patch_args = _patch_namespace(cfg)
    v13.ensure_models(seed_cli, patch_args)
    v13.patch_seedvr2_backend(seed_cli, patch_args)

    import numpy as np
    import torch

    from src.core.generation_utils import (
        setup_generation_context,
        prepare_runner,
        compute_generation_info,
    )
    from src.core.generation_phases import (
        encode_all_batches,
        upscale_all_batches,
        decode_all_batches,
        postprocess_all_batches,
    )
    from src.optimization.memory_manager import (
        manage_model_device,
        clear_memory,
        get_ram_usage,
    )

    debug = seed_cli.debug
    debug.enabled = bool(cfg["debug"])

    base_ctx = setup_generation_context(
        dit_device="cuda:0",
        vae_device="cuda:0",
        dit_offload_device=None,
        vae_offload_device=None,
        tensor_offload_device="cpu",
        debug=debug,
    )

    if phase == "encode":
        dit_cache = False
        vae_cache = True
    elif phase == "dit":
        dit_cache = True
        vae_cache = False
    elif phase == "decode":
        dit_cache = False
        vae_cache = True
    else:
        raise ValueError(f"Unknown phase: {phase}")

    runner, cache_context = prepare_runner(
        dit_model=cfg["dit_model"],
        vae_model=cfg["vae_model"],
        model_dir=cfg["model_dir"],
        debug=debug,
        ctx=base_ctx,
        dit_cache=dit_cache,
        vae_cache=vae_cache,
        dit_id=f"phase_{phase}_dit",
        vae_id=f"phase_{phase}_vae",
        block_swap_config={
            "blocks_to_swap": 0,
            "swap_io_components": False,
            "offload_device": None,
        },
        encode_tiled=True,
        encode_tile_size=(cfg["encode_tile_size"], cfg["encode_tile_size"]),
        encode_tile_overlap=(cfg["encode_tile_overlap"], cfg["encode_tile_overlap"]),
        decode_tiled=True,
        decode_tile_size=(cfg["decode_tile_size"], cfg["decode_tile_size"]),
        decode_tile_overlap=(cfg["decode_tile_overlap"], cfg["decode_tile_overlap"]),
        tile_debug="false",
        attention_mode=cfg["attention_mode"],
        torch_compile_args_dit=None,
        torch_compile_args_vae=None,
    )

    return BackendBundle(
        torch=torch,
        np=np,
        seed_cli=seed_cli,
        debug=debug,
        setup_generation_context=setup_generation_context,
        prepare_runner=prepare_runner,
        compute_generation_info=compute_generation_info,
        encode_all_batches=encode_all_batches,
        upscale_all_batches=upscale_all_batches,
        decode_all_batches=decode_all_batches,
        postprocess_all_batches=postprocess_all_batches,
        manage_model_device=manage_model_device,
        clear_memory=clear_memory,
        get_ram_usage=get_ram_usage,
        runner=runner,
        cache_context=cache_context,
    )


# ---------------------------------------------------------------------------
# Worker-side source loading
# ---------------------------------------------------------------------------

def _stop_ffmpeg_process(proc: subprocess.Popen[Any]) -> None:
    """Terminate and reap a failed per-window FFmpeg child."""
    if proc.poll() is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def _decode_source_window_attempt(
    bundle: BackendBundle,
    cfg: dict[str, Any],
    info: dict[str, Any],
    job: dict[str, Any],
    preroll_frames: int,
) -> list[Any]:
    """
    Decode one source window.

    preroll_frames == 0 preserves the original fast/accurate seek path.

    On retry, seek to an earlier frame and use FFmpeg's frame-based trim filter
    to discard the preroll before conversion to rgb48le. This gives inter-frame
    codecs enough preceding GOP/reference material while keeping the returned
    window aligned to the planned source frame index.
    """
    torch = bundle.torch
    np = bundle.np

    width = int(info["width"])
    height = int(info["height"])
    fps_num = int(info["fps_num"])
    fps_den = int(info["fps_den"])
    fps = Fraction(fps_num, fps_den)

    start_frame = int(job["start_frame"])
    real_count = int(job["real_count"])
    base_frame = max(0, start_frame - int(preroll_frames))
    skip_frames = start_frame - base_frame

    # Use exact rational arithmetic until formatting the timestamp.
    base_seconds = Fraction(base_frame * fps_den, fps_num)
    base_seconds_float = float(base_seconds)
    frame_bytes = width * height * 3 * 2

    ffmpeg = "/usr/sbin/ffmpeg"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "warning",
        "-ss", f"{base_seconds_float:.9f}",
        "-i", cfg["input"],
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
    ]

    if skip_frames:
        # trim=start_frame is deliberately frame-based rather than timestamp-
        # based. After the earlier accurate input seek, discard exactly the
        # number of nominal source frames used as decoder preroll.
        cmd += [
            "-vf",
            f"trim=start_frame={skip_frames},setpts=PTS-STARTPTS",
        ]

    cmd += [
        "-frames:v", str(real_count),
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if proc.stdout is None:
        _stop_ffmpeg_process(proc)
        raise RuntimeError("Could not open FFmpeg decoder stdout")

    frames: list[Any] = []
    failure: Optional[BaseException] = None
    status: Optional[int] = None

    try:
        for frame_index in range(real_count):
            data = proc.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                raise RuntimeError(
                    f"FFmpeg short read for window={job['window_id']} "
                    f"start_frame={start_frame} real_count={real_count} "
                    f"planned_total_frames={info['frames']} "
                    f"seek_base_frame={base_frame} "
                    f"preroll_frames={skip_frames} "
                    f"seek_base_seconds={base_seconds_float:.9f} "
                    f"decoded_before_failure={frame_index}/{real_count}: "
                    f"{len(data)} of {frame_bytes} bytes"
                )

            arr_u16 = np.frombuffer(data, dtype="<u2").reshape(
                height, width, 3
            )
            arr_f16 = (
                arr_u16.astype(np.float32) / 65535.0
            ).astype(np.float16)
            frames.append(torch.from_numpy(arr_f16.copy()))

    except BaseException as exc:
        failure = exc
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass

        if failure is not None:
            _stop_ffmpeg_process(proc)
        else:
            status = proc.wait()

    if failure is not None:
        raise failure

    if status != 0:
        raise RuntimeError(
            f"FFmpeg decoder exited with status {status} for "
            f"window={job['window_id']} base_frame={base_frame} "
            f"preroll_frames={skip_frames}"
        )

    if len(frames) != real_count:
        raise RuntimeError(
            f"FFmpeg decoded {len(frames)} frames for window "
            f"{job['window_id']}; expected {real_count}"
        )

    return frames


def load_source_window(
    bundle: BackendBundle,
    cfg: dict[str, Any],
    info: dict[str, Any],
    job: dict[str, Any],
) -> Any:
    """
    Decode exactly job.real_count source frames and reflect-pad to window_frames.

    The normal path preserves the original per-window FFmpeg accurate seek.
    If that seek fails (notably with some H.264/open-GOP material near EOF),
    retry only the affected window with 5s, 30s, then 120s of decoder preroll.
    """
    torch = bundle.torch

    fps = Fraction(int(info["fps_num"]), int(info["fps_den"]))
    start_frame = int(job["start_frame"])

    retry_frames = [0]
    for seconds in SOURCE_SEEK_PREROLL_RETRIES_SECONDS:
        frames = int(round(float(fps) * seconds))
        frames = min(start_frame, max(0, frames))
        if frames not in retry_frames:
            retry_frames.append(frames)

    frames = None
    last_error: Optional[BaseException] = None

    for attempt_index, preroll_frames in enumerate(retry_frames):
        try:
            frames = _decode_source_window_attempt(
                bundle,
                cfg,
                info,
                job,
                preroll_frames=preroll_frames,
            )
            if attempt_index:
                log(
                    f"[source-retry] window={int(job['window_id']):06d} "
                    f"recovered with {preroll_frames} frames "
                    f"(~{preroll_frames / float(fps):.1f}s) preroll"
                )
            break
        except BaseException as exc:
            last_error = exc
            if attempt_index + 1 >= len(retry_frames):
                break

            next_preroll = retry_frames[attempt_index + 1]
            eprint(
                f"[source-retry] window={int(job['window_id']):06d} "
                f"seek attempt failed: {type(exc).__name__}: {exc}"
            )
            eprint(
                f"[source-retry] retrying with {next_preroll} frames "
                f"(~{next_preroll / float(fps):.1f}s) decoder preroll"
            )

    if frames is None:
        assert last_error is not None
        raise RuntimeError(
            f"All FFmpeg seek strategies failed for window "
            f"{job['window_id']}: {last_error}"
        ) from last_error

    target = int(cfg["window_frames"])
    if len(frames) < target:
        original = list(frames)
        if len(original) == 1:
            while len(frames) < target:
                frames.append(original[0])
        else:
            while len(frames) < target:
                for idx in range(len(original) - 2, -1, -1):
                    frames.append(original[idx])
                    if len(frames) == target:
                        break
                if len(frames) == target:
                    break
                for idx in range(1, len(original)):
                    frames.append(original[idx])
                    if len(frames) == target:
                        break

    frames = frames[:target]
    return torch.stack(frames, dim=0).contiguous()


def fresh_ctx(bundle: BackendBundle) -> dict[str, Any]:
    ctx = bundle.setup_generation_context(
        dit_device="cuda:0",
        vae_device="cuda:0",
        dit_offload_device=None,
        vae_offload_device=None,
        tensor_offload_device="cpu",
        debug=bundle.debug,
    )
    ctx["cache_context"] = bundle.cache_context
    return ctx


# ---------------------------------------------------------------------------
# Worker phase implementations
# ---------------------------------------------------------------------------

def encode_one(
    bundle: BackendBundle,
    cfg: dict[str, Any],
    info: dict[str, Any],
    job: dict[str, Any],
    output_path: Path,
    config_hash: str,
) -> None:
    torch = bundle.torch
    frames = load_source_window(bundle, cfg, info, job)

    ctx = fresh_ctx(bundle)
    frames, _ = bundle.compute_generation_info(
        ctx=ctx,
        images=frames,
        resolution=cfg["resolution"],
        max_resolution=cfg["max_resolution"],
        batch_size=cfg["window_frames"],
        uniform_batch_size=True,
        seed=cfg["seed"],
        prepend_frames=0,
        temporal_overlap=0,
        debug=bundle.debug,
    )

    ctx = bundle.encode_all_batches(
        bundle.runner,
        ctx=ctx,
        images=frames,
        debug=bundle.debug,
        batch_size=cfg["window_frames"],
        uniform_batch_size=True,
        seed=cfg["seed"],
        progress_callback=None,
        temporal_overlap=0,
        resolution=cfg["resolution"],
        max_resolution=cfg["max_resolution"],
        input_noise_scale=cfg["input_noise_scale"],
        color_correction=cfg["color_correction"],
    )

    latent = ctx["all_latents"][0].detach().to(device="cpu").contiguous()
    true_target_dims = tuple(int(x) for x in ctx["true_target_dims"])

    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "config_hash": config_hash,
        "phase": "encoded",
        "job": job,
        "true_target_dims": true_target_dims,
        "latent": latent,
    }
    atomic_torch_save(output_path, payload)

    del payload, latent, ctx, frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dit_one(
    bundle: BackendBundle,
    cfg: dict[str, Any],
    input_path: Path,
    output_path: Path,
    config_hash: str,
) -> None:
    torch = bundle.torch
    source = torch_load_cpu(input_path)
    if source.get("config_hash") != config_hash:
        raise RuntimeError(f"Cache configuration mismatch: {input_path}")

    latent = source["latent"].contiguous()
    job = source["job"]
    true_target_dims = tuple(source["true_target_dims"])

    ctx = fresh_ctx(bundle)
    ctx["all_latents"] = [latent]
    ctx["all_ori_lengths"] = [int(job["real_count"])]
    ctx["true_target_dims"] = true_target_dims
    ctx["total_frames"] = int(job["real_count"])
    ctx["actual_temporal_overlap"] = 0
    ctx["is_rgba"] = False

    ctx = bundle.upscale_all_batches(
        bundle.runner,
        ctx=ctx,
        debug=bundle.debug,
        progress_callback=None,
        seed=cfg["seed"],
        latent_noise_scale=cfg["latent_noise_scale"],
        cache_model=True,
    )

    restored = ctx["all_upscaled_latents"][0].detach().to(device="cpu").contiguous()
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "config_hash": config_hash,
        "phase": "restored",
        "job": job,
        "true_target_dims": true_target_dims,
        "latent": restored,
    }
    atomic_torch_save(output_path, payload)

    del payload, restored, ctx, latent, source
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def postprocess_decoded_window_chunked(
    bundle: BackendBundle,
    cfg: dict[str, Any],
    source_frames: Any,
    decoded_cpu: Any,
    true_target_dims: tuple[int, int],
    real_count: int,
) -> Any:
    """
    Apply Phase 4 in bounded temporal chunks while keeping the decoded window
    itself on CPU.

    Why this exists:
      * decode_all_batches() correctly offloads final_video to CPU.
      * upstream Phase 4 moves a complete batch back to vae_device.
      * LAB then converts both content and reference to float32 and allocates
        additional wavelet/LAB/histogram temporaries.
      * A 65-frame 2880x2160 LAB batch exceeds a 32 GB 5090 once VAE decode
        state/model residency is included.

    The 65-frame VAE/DiT inference window is NOT changed. Only color correction
    and normalization are broken into smaller temporal groups. The default 17f
    group size matches the prior known-good sliding-window correction scale.

    Before Phase 4, move the cached VAE model to CPU and release unused CUDA
    allocator blocks. The next decode job automatically moves the VAE back to
    the worker GPU.
    """
    torch = bundle.torch
    chunk_frames = int(cfg["postprocess_chunk_frames"])

    # SeedVR2 manage_model_device() requires an actual torch.device object;
    # passing the string "cpu" fails because the backend accesses .type.

    if chunk_frames < 1:
        raise RuntimeError("postprocess_chunk_frames must be >= 1")

    # Phase 4 does not use the VAE. Keeping its weights resident on the decoder
    # GPU only steals memory from LAB's float32 working set.
    if bundle.runner.vae is not None:
        bundle.manage_model_device(
            model=bundle.runner.vae,
            target_device=torch.device("cpu"),
            model_name="VAE",
            debug=bundle.debug,
            reason="freeing decoder VRAM before chunked Phase 4 LAB",
            runner=bundle.runner,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for start in range(0, real_count, chunk_frames):
        end = min(real_count, start + chunk_frames)
        count = end - start

        # Build a fresh lightweight context so upstream Phase 4 can reconstruct
        # exactly the corresponding source/reference frames. No model is loaded.
        chunk_ctx = fresh_ctx(bundle)
        chunk_source = source_frames[start:end].contiguous()

        chunk_source, _ = bundle.compute_generation_info(
            ctx=chunk_ctx,
            images=chunk_source,
            resolution=cfg["resolution"],
            max_resolution=cfg["max_resolution"],
            batch_size=chunk_frames,
            uniform_batch_size=False,
            seed=cfg["seed"],
            prepend_frames=0,
            temporal_overlap=0,
            debug=bundle.debug,
        )

        chunk_ctx["input_images"] = chunk_source
        chunk_ctx["final_video"] = decoded_cpu[start:end]
        chunk_ctx["decode_batch_info"] = [(0, count, 0, count)]
        chunk_ctx["all_ori_lengths"] = [count]
        chunk_ctx["true_target_dims"] = true_target_dims
        chunk_ctx["total_frames"] = count
        chunk_ctx["actual_temporal_overlap"] = 0
        chunk_ctx["is_rgba"] = False

        if cfg["color_correction"] != "none":
            chunk_ctx["batch_metadata"] = [(0, count, 0)]

        bundle.postprocess_all_batches(
            ctx=chunk_ctx,
            debug=bundle.debug,
            progress_callback=None,
            color_correction=cfg["color_correction"],
            prepend_frames=0,
            temporal_overlap=0,
            batch_size=chunk_frames,
        )

        # postprocess_all_batches writes back to final_video's CPU storage.
        # Verify that it retained the exact requested chunk length.
        processed = chunk_ctx["final_video"]
        if processed.shape[0] != count:
            raise RuntimeError(
                f"Chunked Phase 4 returned {processed.shape[0]} frames; "
                f"expected {count} for range {start}:{end}"
            )

        # If upstream ever replaces final_video rather than preserving the view,
        # explicitly copy it back into our full CPU window.
        if processed.data_ptr() != decoded_cpu[start:end].data_ptr():
            decoded_cpu[start:end].copy_(
                processed.to(device="cpu", dtype=decoded_cpu.dtype)
            )

        del processed, chunk_ctx, chunk_source

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return decoded_cpu



def decode_one(
    bundle: BackendBundle,
    cfg: dict[str, Any],
    info: dict[str, Any],
    input_path: Path,
    output_path: Path,
    config_hash: str,
) -> None:
    torch = bundle.torch
    source = torch_load_cpu(input_path)
    if source.get("config_hash") != config_hash:
        raise RuntimeError(f"Cache configuration mismatch: {input_path}")

    job = source["job"]
    restored = source["latent"].contiguous()
    true_target_dims = tuple(source["true_target_dims"])

    # Re-read the exact source window for LAB color correction.
    frames = load_source_window(bundle, cfg, info, job)

    ctx = fresh_ctx(bundle)
    frames, _ = bundle.compute_generation_info(
        ctx=ctx,
        images=frames,
        resolution=cfg["resolution"],
        max_resolution=cfg["max_resolution"],
        batch_size=cfg["window_frames"],
        uniform_batch_size=True,
        seed=cfg["seed"],
        prepend_frames=0,
        temporal_overlap=0,
        debug=bundle.debug,
    )

    # Phase-split reconstruction of the state normally produced by encode.
    ctx["input_images"] = frames
    ctx["all_upscaled_latents"] = [restored]
    ctx["all_ori_lengths"] = [int(job["real_count"])]
    ctx["true_target_dims"] = true_target_dims
    ctx["total_frames"] = int(job["real_count"])
    ctx["actual_temporal_overlap"] = 0
    ctx["is_rgba"] = False
    if cfg["color_correction"] != "none":
        # The original encode saw a full externally-padded 65-frame tensor.
        # Postprocess reconstructs all 65 and trims to the decoded real_count.
        ctx["batch_metadata"] = [(0, cfg["window_frames"], 0)]

    ctx = bundle.decode_all_batches(
        bundle.runner,
        ctx=ctx,
        debug=bundle.debug,
        progress_callback=None,
        cache_model=True,
    )

    # decode_all_batches places final_video on tensor_offload_device (CPU).
    # Preserve that CPU tensor and perform Phase 4 in bounded temporal groups
    # after explicitly offloading the VAE model from the decoder GPU.
    final_video = (
        ctx["final_video"]
        .detach()
        .to(device="cpu", dtype=torch.float16)
        .contiguous()
    )
    final_video = postprocess_decoded_window_chunked(
        bundle=bundle,
        cfg=cfg,
        source_frames=frames,
        decoded_cpu=final_video,
        true_target_dims=true_target_dims,
        real_count=int(job["real_count"]),
    )
    if final_video.shape[0] != int(job["real_count"]):
        raise RuntimeError(
            f"Decoded window {job['window_id']} has {final_video.shape[0]} frames; "
            f"expected {job['real_count']}"
        )

    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "config_hash": config_hash,
        "phase": "decoded",
        "job": job,
        "frames": final_video,
    }
    atomic_torch_save(output_path, payload)

    del payload, final_video, ctx, frames, restored, source
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multiprocessing worker entry points
# ---------------------------------------------------------------------------

def cleanup_worker_memory(
    bundle: BackendBundle,
    phase: str,
    physical_gpu: str,
    window_id: int,
    jobs_completed: int,
) -> None:
    """
    Release per-window CPU/GPU allocator state while preserving cached models.

    SeedVR2's official streaming CLI performs clear_memory(deep=True, force=True)
    between chunks. Deep cleanup does:
      * torch CUDA cache + IPC cache cleanup
      * full generation-2 Python garbage collection
      * malloc_trim(0) on Linux so free glibc arenas return to the OS

    It does not destroy bundle.runner or bundle.cache_context, so the VAE/DiT
    model cache remains warm for the next window.
    """
    try:
        bundle.clear_memory(
            debug=bundle.debug,
            deep=True,
            force=True,
            timer_name=f"phase_split_{phase}_gpu{physical_gpu}",
        )
    except Exception as exc:
        # Cleanup must never turn a successfully processed window into a failed
        # job. The old CUDA-only cleanup is retained as a safe fallback.
        eprint(
            f"[worker-cleanup] WARNING phase={phase} gpu={physical_gpu} "
            f"window={window_id:06d}: {type(exc).__name__}: {exc}"
        )
        try:
            if bundle.torch.cuda.is_available():
                bundle.torch.cuda.empty_cache()
                bundle.torch.cuda.ipc_collect()
        except Exception:
            pass

    if (
        jobs_completed == 1
        or jobs_completed % WORKER_MEMORY_LOG_INTERVAL == 0
    ):
        try:
            process_gib, available_gib, _, _ = bundle.get_ram_usage(
                debug=None
            )
            log(
                f"[worker-memory] phase={phase} gpu={physical_gpu} "
                f"jobs={jobs_completed} rss={process_gib:.2f} GiB "
                f"host-available={available_gib:.1f} GiB"
            )
        except Exception:
            pass


def phase_worker(
    phase: str,
    physical_gpu: str,
    cfg: dict[str, Any],
    info: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
    config_hash: str,
) -> None:
    try:
        bundle = bootstrap_backend(cfg, physical_gpu, phase)
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "worker_init_failed",
                "phase": phase,
                "gpu": physical_gpu,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    jobs_completed = 0

    while True:
        task = task_queue.get()
        if task is None:
            return

        window_id = int(task["job"]["window_id"])
        result_queue.put(
            {
                "kind": "started",
                "phase": phase,
                "gpu": physical_gpu,
                "window_id": window_id,
                "task": task,
            }
        )
        started = time.monotonic()
        try:
            if phase == "encode":
                encode_one(
                    bundle,
                    cfg,
                    info,
                    task["job"],
                    Path(task["output"]),
                    config_hash,
                )
            elif phase == "dit":
                dit_one(
                    bundle,
                    cfg,
                    Path(task["input"]),
                    Path(task["output"]),
                    config_hash,
                )
            elif phase == "decode":
                decode_one(
                    bundle,
                    cfg,
                    info,
                    Path(task["input"]),
                    Path(task["output"]),
                    config_hash,
                )
            else:
                raise RuntimeError(f"Unknown phase {phase}")

            jobs_completed += 1
            cleanup_worker_memory(
                bundle,
                phase,
                physical_gpu,
                window_id,
                jobs_completed,
            )

            result_queue.put(
                {
                    "kind": "done",
                    "phase": phase,
                    "gpu": physical_gpu,
                    "window_id": window_id,
                    "seconds": time.monotonic() - started,
                    "output": task["output"],
                }
            )
        except BaseException as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            low = error_text.lower()
            oom = "out of memory" in low or "cuda error: memory allocation" in low
            traceback_text = traceback.format_exc()

            # The process will retire after reporting the failure, but perform
            # cleanup first so any buffered allocator state is released
            # immediately rather than waiting for process teardown.
            cleanup_worker_memory(
                bundle,
                phase,
                physical_gpu,
                window_id,
                jobs_completed,
            )

            result_queue.put(
                {
                    "kind": "failed",
                    "phase": phase,
                    "gpu": physical_gpu,
                    "window_id": window_id,
                    "seconds": time.monotonic() - started,
                    "output": task["output"],
                    "error": error_text,
                    "oom": oom,
                    "traceback": traceback_text,
                    "task": task,
                }
            )
            # A failed phase can leave backend/library state unreliable. Retire
            # this worker and let the controller requeue its failed job.
            return


# ---------------------------------------------------------------------------
# Stage controllers
# ---------------------------------------------------------------------------

def start_workers(
    mpctx: Any,
    *,
    phase: str,
    gpus: list[str],
    cfg_dict: dict[str, Any],
    info_dict: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
    config_hash: str,
) -> dict[str, Any]:
    workers: dict[str, Any] = {}
    for gpu in gpus:
        process = mpctx.Process(
            target=phase_worker,
            name=f"{phase}-gpu{gpu}",
            args=(
                phase,
                gpu,
                cfg_dict,
                info_dict,
                task_queue,
                result_queue,
                config_hash,
            ),
        )
        process.start()
        workers[gpu] = process
    return workers


def close_mp_queue(q: Any) -> None:
    """
    Explicitly close a multiprocessing.Queue owned by the parent process.

    Leaving Queue feeder threads to interpreter shutdown can make a completed
    long-running job appear to hang after printing its final status.
    """
    if q is None:
        return
    try:
        q.close()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        q.join_thread()
    except (AttributeError, OSError, ValueError):
        pass


def stop_workers(task_queue: Any, workers: dict[str, Any]) -> None:
    """
    Stop and reap every worker process.

    Workers get a normal sentinel first. Any process that does not exit within
    the grace period is terminated, then killed as a last resort.
    """
    alive = [p for p in workers.values() if p.is_alive()]
    for _ in alive:
        try:
            task_queue.put(None)
        except (OSError, ValueError):
            break

    for process in workers.values():
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def cleanup_child_processes() -> None:
    """
    Final safety net before returning control to the user's shell.
    """
    for process in mp.active_children():
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)


def capture_terminal_state() -> Any:
    """Capture the invoking terminal state so child libraries cannot leave it raw."""
    if not sys.stdin.isatty():
        return None
    try:
        import termios
        return (sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno()))
    except (ImportError, OSError, termios.error):
        return None


def restore_terminal_state(state: Any) -> None:
    """Restore terminal echo/canonical mode before returning to Bash."""
    if state is None:
        return
    try:
        import termios
        fd, attrs = state
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except (ImportError, OSError, termios.error):
        pass


def stage_encode(
    cfg: Config,
    info: VideoInfo,
    jobs: list[WindowJob],
    encoded_dir: Path,
    config_hash: str,
) -> None:
    log("\n=== Stage A: four-GPU VAE pre-encode ===")
    for gpu in cfg.encode_gpus:
        minimum = cfg.encode_min_free_mib.get(gpu, 0)
        check_gpu(gpu, minimum)

    missing = [
        job for job in jobs
        if not (encoded_dir / f"{job.name}.pt").is_file()
    ]
    if not missing:
        log("All encoded latent files already exist; skipping Stage A.")
        return

    log(f"Encoding {len(missing)} missing windows with GPUs {','.join(cfg.encode_gpus)}")
    mpctx = mp.get_context("spawn")
    task_q = mpctx.Queue()
    result_q = mpctx.Queue()

    cfg_dict = asdict(cfg)
    info_dict = asdict(info)
    workers = start_workers(
        mpctx,
        phase="encode",
        gpus=cfg.encode_gpus,
        cfg_dict=cfg_dict,
        info_dict=info_dict,
        task_queue=task_q,
        result_queue=result_q,
        config_hash=config_hash,
    )

    jobs_by_id = {job.window_id: job for job in missing}
    pending_ids = set(jobs_by_id)
    attempts: dict[int, int] = {job.window_id: 0 for job in missing}

    for job in missing:
        task_q.put(
            {
                "job": asdict(job),
                "output": str(encoded_dir / f"{job.name}.pt"),
            }
        )

    completed = 0
    retired: set[str] = set()
    running_by_gpu: dict[str, dict[str, Any]] = {}
    try:
        while pending_ids:
            try:
                result = result_q.get(timeout=5)
            except queue.Empty:
                dead = [gpu for gpu, proc in workers.items() if not proc.is_alive() and gpu not in retired]
                for gpu in dead:
                    retired.add(gpu)
                    lost = running_by_gpu.pop(gpu, None)
                    if lost is not None:
                        wid = int(lost["job"]["window_id"])
                        out = Path(lost["output"])
                        if out.is_file():
                            pending_ids.discard(wid)
                            completed += 1
                            log(
                                f"[encode] recovered completed window={wid:06d} "
                                f"after GPU {gpu} exited before reporting"
                            )
                        elif wid in pending_ids:
                            attempts[wid] += 1
                            if attempts[wid] > 2:
                                die(f"Encode window {wid} repeatedly lost with worker failure")
                            eprint(
                                f"[encode] GPU {gpu} exited while processing "
                                f"window {wid:06d}; requeuing."
                            )
                            task_q.put(lost)
                if len(retired) == len(workers):
                    die("All encode workers exited before Stage A completed")
                continue

            kind = result["kind"]
            gpu = result["gpu"]

            if kind == "started":
                running_by_gpu[gpu] = result["task"]
                continue

            if kind == "worker_init_failed":
                retired.add(gpu)
                eprint(f"Encode worker GPU {gpu} failed initialization: {result['error']}")
                eprint(result["traceback"])
                if len(retired) == len(workers):
                    die("No encode workers remain")
                continue

            wid = int(result["window_id"])
            running_by_gpu.pop(gpu, None)
            if kind == "done":
                if wid in pending_ids:
                    pending_ids.remove(wid)
                    completed += 1
                log(
                    f"[encode] {completed}/{len(missing)} "
                    f"window={wid:06d} gpu={gpu} {result['seconds']:.1f}s"
                )
                continue

            if kind == "failed":
                retired.add(gpu)
                eprint(
                    f"[encode] GPU {gpu} failed window {wid:06d}: "
                    f"{result['error']}"
                )
                if result.get("oom"):
                    eprint(f"[encode] Retiring GPU {gpu} after CUDA OOM and requeuing the window.")
                else:
                    eprint(result["traceback"])

                attempts[wid] += 1
                if attempts[wid] > 2:
                    die(f"Encode window {wid} failed more than twice")
                task_q.put(result["task"])

                if len(retired) == len(workers):
                    die("All encode workers have failed")
    finally:
        stop_workers(task_q, workers)
        close_mp_queue(task_q)
        close_mp_queue(result_q)


def stage_dit(
    cfg: Config,
    jobs: list[WindowJob],
    encoded_dir: Path,
    restored_dir: Path,
    config_hash: str,
) -> None:
    log("\n=== Stage B: RTX PRO 6000 DiT ===")
    check_gpu(cfg.dit_gpu, cfg.dit_min_free_mib, expected="RTX PRO 6000")

    missing = [
        job for job in jobs
        if not (restored_dir / f"{job.name}.pt").is_file()
    ]
    if not missing:
        log("All restored latent files already exist; skipping Stage B.")
        return

    mpctx = mp.get_context("spawn")
    task_q = mpctx.Queue()
    result_q = mpctx.Queue()
    worker = start_workers(
        mpctx,
        phase="dit",
        gpus=[cfg.dit_gpu],
        cfg_dict=asdict(cfg),
        info_dict={},
        task_queue=task_q,
        result_queue=result_q,
        config_hash=config_hash,
    )

    for job in missing:
        encoded = encoded_dir / f"{job.name}.pt"
        if not encoded.is_file():
            die(f"Encoded latent missing for DiT: {encoded}")
        task_q.put(
            {
                "job": asdict(job),
                "input": str(encoded),
                "output": str(restored_dir / f"{job.name}.pt"),
            }
        )

    done = 0
    try:
        while done < len(missing):
            try:
                result = result_q.get(timeout=5)
            except queue.Empty:
                proc = worker[cfg.dit_gpu]
                if not proc.is_alive():
                    die("DiT worker exited unexpectedly")
                continue

            if result["kind"] == "started":
                continue
            if result["kind"] == "worker_init_failed":
                eprint(result["traceback"])
                die(f"DiT worker initialization failed: {result['error']}")
            if result["kind"] == "failed":
                eprint(result["traceback"])
                die(
                    f"DiT failed window {result['window_id']}: "
                    f"{result['error']}"
                )
            if result["kind"] != "done":
                die(f"Unexpected DiT worker event: {result['kind']}")

            done += 1
            log(
                f"[dit] {done}/{len(missing)} "
                f"window={result['window_id']:06d} "
                f"{result['seconds']:.1f}s"
            )
    finally:
        stop_workers(task_q, worker)
        close_mp_queue(task_q)
        close_mp_queue(result_q)


class NVENCRGB48Writer:
    """Stream RGB float frames to 10-bit HEVC NVENC on a physical GPU."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps_text: str,
        gpu: str,
        preset: str,
        cq: int,
    ) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.frames_written = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = require_program("/usr/sbin/ffmpeg")

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "-video_size", f"{width}x{height}",
            "-framerate", fps_text,
            "-i", "pipe:0",
            "-an",
            "-c:v", "hevc_nvenc",
            "-gpu", str(gpu),
            "-preset", preset,
            "-tune", "hq",
            "-profile:v", "main10",
            "-rc:v", "vbr",
            "-cq:v", str(cq),
            "-b:v", "0",
            "-pix_fmt", "p010le",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-color_range", "tv",
            "-fps_mode", "cfr",
            str(path),
        ]
        eprint("+ " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError("Could not open FFmpeg encoder stdin")

    def write_tensor(self, tensor: Any) -> None:
        import numpy as np

        for frame in tensor:
            if self.proc.poll() is not None:
                raise RuntimeError("FFmpeg NVENC process terminated early")
            arr = frame.detach().to(device="cpu").float().clamp_(0.0, 1.0).numpy()
            arr_u16 = np.rint(arr * 65535.0).astype("<u2", copy=False)
            assert self.proc.stdin is not None
            self.proc.stdin.write(memoryview(arr_u16).cast("B"))
            self.frames_written += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        status = self.proc.wait()
        if status != 0:
            raise RuntimeError(f"FFmpeg NVENC exited with status {status}")

    def terminate(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class WindowAssembler:
    """
    Assemble independently processed atomic windows with v13's overlap policy.
    """

    def __init__(self, writer: NVENCRGB48Writer, overlap: int, window: int) -> None:
        self.writer = writer
        self.overlap = overlap
        self.window = window
        self.hop = window - overlap
        self.pending_tail = None
        self.current_shot: Optional[int] = None
        self.written = 0

    def _blend(self, previous: Any, current: Any) -> Any:
        import torch

        if self.overlap == 0:
            return current[:0]
        blended = []
        for index in range(self.overlap):
            alpha = 0.5 - 0.5 * math.cos(
                math.pi * (index + 1) / (self.overlap + 1)
            )
            frame = (
                previous[index].float().mul(1.0 - alpha)
                .add(current[index].float(), alpha=alpha)
            )
            blended.append(frame)
        return torch.stack(blended, dim=0)

    def consume(self, job: WindowJob, frames: Any) -> None:
        if frames.shape[0] != job.real_count:
            raise RuntimeError(
                f"Assembler got {frames.shape[0]} frames for window "
                f"{job.window_id}; expected {job.real_count}"
            )

        if job.is_first:
            self.current_shot = job.shot_id
            self.pending_tail = None
            if job.is_last:
                self.writer.write_tensor(frames[:job.real_count])
                self.written += job.real_count
                return

            self.writer.write_tensor(frames[:self.hop])
            self.written += self.hop
            # First full window must have all overlap-tail frames.
            self.pending_tail = frames[self.hop:self.window].clone()
            return

        if self.current_shot != job.shot_id or self.pending_tail is None:
            raise RuntimeError(
                f"Assembler state mismatch at window {job.window_id} "
                f"(shot {job.shot_id})"
            )

        head = frames[:self.overlap]
        blended = self._blend(self.pending_tail, head)
        self.writer.write_tensor(blended)
        self.written += self.overlap
        del blended

        if job.is_last:
            self.writer.write_tensor(frames[self.overlap:job.real_count])
            self.written += job.real_count - self.overlap
            self.pending_tail = None
            return

        middle_end = self.window - self.overlap
        self.writer.write_tensor(frames[self.overlap:middle_end])
        self.written += middle_end - self.overlap
        self.pending_tail = frames[middle_end:self.window].clone()


def remux_original_streams(video_only: Path, source: Path, output: Path) -> None:
    ffmpeg = require_program("/usr/sbin/ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-i", str(video_only),
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-map", "1:d?",
        "-map", "1:t?",
        "-map_metadata", "1",
        "-map_chapters", "1",
        "-c", "copy",
        "-disposition:v:0", "default",
        str(output),
    ]
    eprint("+ " + " ".join(cmd))
    status = subprocess.run(cmd).returncode
    if status != 0:
        die(f"Final remux failed with status {status}")


def stage_decode_and_assemble(
    cfg: Config,
    info: VideoInfo,
    jobs: list[WindowJob],
    restored_dir: Path,
    decoded_dir: Path,
    final_video_only: Path,
    config_hash: str,
) -> None:
    log("\n=== Stage C: three-GPU VAE decode + ordered assembly ===")
    for gpu in cfg.decode_gpus:
        minimum = cfg.decode_min_free_mib.get(gpu, 0)
        expected = "RTX 5090" if gpu in {"0", "1"} else None
        check_gpu(gpu, minimum, expected=expected)
    check_gpu(cfg.nvenc_gpu, 1000)

    # Stage C assembly is intentionally restarted from frame zero. Old decoded
    # windows are not trusted because they may belong to an interrupted writer.
    if decoded_dir.exists() and not cfg.keep_decoded:
        shutil.rmtree(decoded_dir)
    decoded_dir.mkdir(parents=True, exist_ok=True)
    final_video_only.unlink(missing_ok=True)

    for job in jobs:
        restored = restored_dir / f"{job.name}.pt"
        if not restored.is_file():
            die(f"Restored latent missing for decode: {restored}")

    # Target dimensions are stored in each latent. Load only the first metadata.
    first_payload = torch_load_cpu(restored_dir / f"{jobs[0].name}.pt")
    true_h, true_w = [int(x) for x in first_payload["true_target_dims"]]
    del first_payload

    writer = NVENCRGB48Writer(
        final_video_only,
        width=true_w,
        height=true_h,
        fps_text=info.fps_text,
        gpu=cfg.nvenc_gpu,
        preset=cfg.nvenc_preset,
        cq=cfg.nvenc_cq,
    )
    assembler = WindowAssembler(writer, cfg.overlap_frames, cfg.window_frames)

    mpctx = mp.get_context("spawn")
    task_q = mpctx.Queue()
    result_q = mpctx.Queue()
    workers = start_workers(
        mpctx,
        phase="decode",
        gpus=cfg.decode_gpus,
        cfg_dict=asdict(cfg),
        info_dict=asdict(info),
        task_queue=task_q,
        result_queue=result_q,
        config_hash=config_hash,
    )

    jobs_by_id = {job.window_id: job for job in jobs}
    next_submit = 0
    next_assemble = 0
    inflight = 0
    completed_files: dict[int, Path] = {}
    active_gpus = set(cfg.decode_gpus)
    attempts: dict[int, int] = {job.window_id: 0 for job in jobs}
    running_by_gpu: dict[str, dict[str, Any]] = {}

    def submit(job: WindowJob) -> None:
        nonlocal inflight
        task_q.put(
            {
                "job": asdict(job),
                "input": str(restored_dir / f"{job.name}.pt"),
                "output": str(decoded_dir / f"{job.name}.pt"),
            }
        )
        inflight += 1

    try:
        while next_submit < len(jobs) and inflight < cfg.decode_inflight:
            submit(jobs[next_submit])
            next_submit += 1

        done_count = 0
        while done_count < len(jobs):
            try:
                result = result_q.get(timeout=5)
            except queue.Empty:
                dead = [
                    gpu for gpu, proc in workers.items()
                    if not proc.is_alive() and gpu in active_gpus
                ]
                for gpu in dead:
                    active_gpus.remove(gpu)
                    lost = running_by_gpu.pop(gpu, None)
                    if lost is not None:
                        wid = int(lost["job"]["window_id"])
                        out = Path(lost["output"])
                        if out.is_file():
                            completed_files[wid] = out
                            done_count += 1
                            inflight -= 1
                            log(
                                f"[decode] recovered completed window={wid:06d} "
                                f"after GPU {gpu} exited before reporting"
                            )
                        else:
                            attempts[wid] += 1
                            if attempts[wid] > 2:
                                die(f"Decode window {wid} repeatedly lost with worker failure")
                            eprint(
                                f"[decode] GPU {gpu} exited while processing "
                                f"window {wid:06d}; requeuing."
                            )
                            task_q.put(lost)
                if not active_gpus and inflight > 0:
                    die("All decode workers exited before Stage C completed")
                continue

            kind = result["kind"]
            gpu = result["gpu"]

            if kind == "started":
                running_by_gpu[gpu] = result["task"]
                continue

            if kind == "worker_init_failed":
                active_gpus.discard(gpu)
                eprint(f"Decode worker GPU {gpu} failed initialization: {result['error']}")
                eprint(result["traceback"])
                if not active_gpus:
                    die("No decode workers remain")
                continue

            inflight -= 1
            wid = int(result["window_id"])
            running_by_gpu.pop(gpu, None)

            if kind == "failed":
                active_gpus.discard(gpu)
                attempts[wid] += 1
                eprint(
                    f"[decode] GPU {gpu} failed window {wid:06d}: "
                    f"{result['error']}"
                )
                if result.get("oom"):
                    eprint(
                        f"[decode] Retiring GPU {gpu}; window {wid:06d} "
                        f"will be requeued to the remaining decode workers."
                    )
                else:
                    eprint(result["traceback"])

                if attempts[wid] > 2:
                    die(f"Decode window {wid} failed more than twice")
                if not active_gpus:
                    die("All decode workers have failed")
                task_q.put(result["task"])
                inflight += 1
            else:
                completed_files[wid] = Path(result["output"])
                done_count += 1
                log(
                    f"[decode] {done_count}/{len(jobs)} "
                    f"window={wid:06d} gpu={gpu} {result['seconds']:.1f}s"
                )

            # Assemble every now-contiguous completed window.
            while next_assemble in completed_files:
                path = completed_files.pop(next_assemble)
                payload = torch_load_cpu(path)
                if payload.get("config_hash") != config_hash:
                    die(f"Decoded cache hash mismatch: {path}")
                job = jobs_by_id[next_assemble]
                assembler.consume(job, payload["frames"])
                del payload
                if not cfg.keep_decoded:
                    path.unlink(missing_ok=True)
                next_assemble += 1

            # Keep only a bounded number of decoded windows in flight/on disk.
            while (
                next_submit < len(jobs)
                and inflight + len(completed_files) < cfg.decode_inflight
                and active_gpus
            ):
                submit(jobs[next_submit])
                next_submit += 1

        if next_assemble != len(jobs):
            die(
                f"Decode finished but assembler stopped at {next_assemble}/"
                f"{len(jobs)} windows"
            )

        writer.close()
    except BaseException:
        writer.terminate()
        raise
    finally:
        stop_workers(task_q, workers)
        close_mp_queue(task_q)
        close_mp_queue(result_q)

    expected = expected_output_frames(jobs, cfg.overlap_frames)
    if assembler.written != expected:
        die(
            f"Final video frame mismatch: wrote {assembler.written}, "
            f"expected {expected}"
        )
    log(f"Stage C wrote {assembler.written} frames.")



def stage_dit_decode_concurrent(
    cfg: Config,
    info: VideoInfo,
    jobs: list[WindowJob],
    encoded_dir: Path,
    restored_dir: Path,
    decoded_dir: Path,
    final_video_only: Path,
    config_hash: str,
) -> None:
    """
    Run the 6000 DiT producer concurrently with the three VAE decode consumers.

    Restored latent files are the durable producer/consumer boundary. Any file
    already present from an interrupted run is immediately eligible for decode;
    the DiT worker receives only missing windows.

    Decode submission is intentionally kept near source order. That bounds
    decoded-frame storage while still allowing the heterogeneous decode workers
    to operate independently.
    """
    log("\n=== Concurrent Stage B + Stage C pipeline ===")
    log("\n--- Stage B: RTX PRO 6000 DiT producer ---")
    check_gpu(cfg.dit_gpu, cfg.dit_min_free_mib, expected="RTX PRO 6000")

    log("\n--- Stage C: multi-GPU VAE decode consumers ---")
    for gpu in cfg.decode_gpus:
        minimum = cfg.decode_min_free_mib.get(gpu, 0)
        expected = "RTX 5090" if gpu in {"0", "1"} else None
        check_gpu(gpu, minimum, expected=expected)

    log("\n--- Output encoder ---")
    if cfg.nvenc_gpu in cfg.decode_gpus:
        name, total, free = gpu_query(cfg.nvenc_gpu)
        log(
            f"  GPU {cfg.nvenc_gpu}: {name} | NVENC output "
            f"(also a Stage C decode worker; {free}/{total} MiB free)"
        )
    elif cfg.nvenc_gpu == cfg.dit_gpu:
        name, total, free = gpu_query(cfg.nvenc_gpu)
        log(
            f"  GPU {cfg.nvenc_gpu}: {name} | NVENC output "
            f"(also the Stage B DiT worker; {free}/{total} MiB free)"
        )
    else:
        check_gpu(cfg.nvenc_gpu, 1000)

    # Every encoded input must already exist: Stage A remains a complete
    # pre-encode pass by design.
    for job in jobs:
        encoded = encoded_dir / f"{job.name}.pt"
        if not encoded.is_file():
            die(f"Encoded latent missing before Stage B+C: {encoded}")

    existing_restored = {
        job.window_id
        for job in jobs
        if (restored_dir / f"{job.name}.pt").is_file()
    }
    missing_dit = [
        job for job in jobs
        if job.window_id not in existing_restored
    ]

    log(
        f"Restored cache at startup: {len(existing_restored)}/{len(jobs)} "
        f"windows complete; DiT has {len(missing_dit)} windows remaining."
    )

    # Assembly is restarted from frame zero, but expensive latent work is not.
    if decoded_dir.exists() and not cfg.keep_decoded:
        shutil.rmtree(decoded_dir)
    decoded_dir.mkdir(parents=True, exist_ok=True)
    final_video_only.unlink(missing_ok=True)

    # Geometry is available in either cache generation.
    metadata_path = None
    for job in jobs:
        candidate = restored_dir / f"{job.name}.pt"
        if candidate.is_file():
            metadata_path = candidate
            break
    if metadata_path is None:
        metadata_path = encoded_dir / f"{jobs[0].name}.pt"

    first_payload = torch_load_cpu(metadata_path)
    true_h, true_w = [int(x) for x in first_payload["true_target_dims"]]
    del first_payload

    writer = NVENCRGB48Writer(
        final_video_only,
        width=true_w,
        height=true_h,
        fps_text=info.fps_text,
        gpu=cfg.nvenc_gpu,
        preset=cfg.nvenc_preset,
        cq=cfg.nvenc_cq,
    )
    assembler = WindowAssembler(writer, cfg.overlap_frames, cfg.window_frames)

    mpctx = mp.get_context("spawn")
    result_q = mpctx.Queue()
    decode_task_q = mpctx.Queue()

    decode_workers = start_workers(
        mpctx,
        phase="decode",
        gpus=cfg.decode_gpus,
        cfg_dict=asdict(cfg),
        info_dict=asdict(info),
        task_queue=decode_task_q,
        result_queue=result_q,
        config_hash=config_hash,
    )

    dit_task_q = None
    dit_workers: dict[str, Any] = {}
    if missing_dit:
        dit_task_q = mpctx.Queue()
        dit_workers = start_workers(
            mpctx,
            phase="dit",
            gpus=[cfg.dit_gpu],
            cfg_dict=asdict(cfg),
            info_dict={},
            task_queue=dit_task_q,
            result_queue=result_q,
            config_hash=config_hash,
        )
        for job in missing_dit:
            dit_task_q.put(
                {
                    "job": asdict(job),
                    "input": str(encoded_dir / f"{job.name}.pt"),
                    "output": str(restored_dir / f"{job.name}.pt"),
                }
            )

    jobs_by_id = {job.window_id: job for job in jobs}
    ready_restored = set(existing_restored)
    decoded_submitted: set[int] = set()
    completed_files: dict[int, Path] = {}
    next_decode_submit = 0
    next_assemble = 0
    decode_inflight = 0
    decode_done = 0
    dit_done_new = 0

    active_decode_gpus = set(cfg.decode_gpus)
    decode_attempts: dict[int, int] = {job.window_id: 0 for job in jobs}
    decode_running_by_gpu: dict[str, dict[str, Any]] = {}
    dit_running: Optional[dict[str, Any]] = None
    dit_finished = not missing_dit
    dit_stop_sent = not missing_dit

    def submit_ready_decodes() -> None:
        nonlocal next_decode_submit, decode_inflight

        while (
            next_decode_submit < len(jobs)
            and decode_inflight + len(completed_files) < cfg.decode_inflight
            and active_decode_gpus
        ):
            job = jobs[next_decode_submit]
            wid = job.window_id

            # Keep decode scheduling close to timeline order. The single DiT
            # producer normally makes this contiguous automatically.
            if wid not in ready_restored:
                break

            if wid not in decoded_submitted:
                decode_task_q.put(
                    {
                        "job": asdict(job),
                        "input": str(restored_dir / f"{job.name}.pt"),
                        "output": str(decoded_dir / f"{job.name}.pt"),
                    }
                )
                decoded_submitted.add(wid)
                decode_inflight += 1
            next_decode_submit += 1

    # Existing restored prefix can start decoding immediately, before the first
    # new DiT window completes.
    submit_ready_decodes()

    try:
        while decode_done < len(jobs):
            try:
                result = result_q.get(timeout=5)
            except queue.Empty:
                # Detect unexpectedly dead decode workers and recover their
                # in-flight task when possible.
                for gpu, proc in decode_workers.items():
                    if gpu not in active_decode_gpus or proc.is_alive():
                        continue
                    active_decode_gpus.remove(gpu)
                    lost = decode_running_by_gpu.pop(gpu, None)
                    if lost is not None:
                        wid = int(lost["job"]["window_id"])
                        out = Path(lost["output"])
                        decode_inflight -= 1
                        if out.is_file():
                            completed_files[wid] = out
                            decode_done += 1
                            log(
                                f"[decode] recovered completed window={wid:06d} "
                                f"after GPU {gpu} exited before reporting"
                            )
                        else:
                            decode_attempts[wid] += 1
                            if decode_attempts[wid] > 2:
                                die(
                                    f"Decode window {wid} repeatedly lost "
                                    "with worker failure"
                                )
                            decoded_submitted.discard(wid)
                            eprint(
                                f"[decode] GPU {gpu} exited while processing "
                                f"window {wid:06d}; requeuing."
                            )
                            decode_task_q.put(lost)
                            decoded_submitted.add(wid)
                            decode_inflight += 1

                if not active_decode_gpus and decode_done < len(jobs):
                    die("All decode workers exited before Stage B+C completed")

                if dit_workers and not dit_finished:
                    proc = dit_workers[cfg.dit_gpu]
                    if not proc.is_alive():
                        if dit_running is not None:
                            wid = int(dit_running["job"]["window_id"])
                            out = Path(dit_running["output"])
                            if out.is_file():
                                ready_restored.add(wid)
                                dit_done_new += 1
                                log(
                                    f"[dit] recovered completed window={wid:06d} "
                                    "after worker exited before reporting"
                                )
                                dit_running = None
                                submit_ready_decodes()
                            else:
                                die(
                                    f"DiT worker exited while processing "
                                    f"window {wid:06d}; rerun will reuse all "
                                    "completed restored latents."
                                )
                        else:
                            die("DiT worker exited unexpectedly")

                continue

            phase = result.get("phase")
            kind = result["kind"]
            gpu = result["gpu"]

            if phase == "dit":
                if kind == "started":
                    dit_running = result["task"]
                    continue
                if kind == "worker_init_failed":
                    eprint(result["traceback"])
                    die(f"DiT worker initialization failed: {result['error']}")
                if kind == "failed":
                    eprint(result["traceback"])
                    die(
                        f"DiT failed window {result['window_id']}: "
                        f"{result['error']}"
                    )
                if kind != "done":
                    die(f"Unexpected DiT worker event: {kind}")

                wid = int(result["window_id"])
                dit_running = None
                ready_restored.add(wid)
                dit_done_new += 1
                total_restored = len(existing_restored) + dit_done_new
                log(
                    f"[dit] {total_restored}/{len(jobs)} "
                    f"window={wid:06d} {result['seconds']:.1f}s "
                    "-> decode eligible"
                )

                if dit_done_new == len(missing_dit):
                    dit_finished = True
                    if dit_task_q is not None and not dit_stop_sent:
                        dit_task_q.put(None)
                        dit_stop_sent = True
                    log(
                        "DiT producer complete; decode/assembly will continue "
                        "until all windows are written."
                    )

                submit_ready_decodes()
                continue

            if phase != "decode":
                die(f"Unexpected worker phase in Stage B+C: {phase!r}")

            if kind == "started":
                decode_running_by_gpu[gpu] = result["task"]
                continue

            if kind == "worker_init_failed":
                active_decode_gpus.discard(gpu)
                eprint(
                    f"Decode worker GPU {gpu} failed initialization: "
                    f"{result['error']}"
                )
                eprint(result["traceback"])
                if not active_decode_gpus:
                    die("No decode workers remain")
                continue

            wid = int(result["window_id"])
            decode_running_by_gpu.pop(gpu, None)
            decode_inflight -= 1

            if kind == "failed":
                active_decode_gpus.discard(gpu)
                decode_attempts[wid] += 1
                eprint(
                    f"[decode] GPU {gpu} failed window {wid:06d}: "
                    f"{result['error']}"
                )
                if result.get("oom"):
                    eprint(
                        f"[decode] Retiring GPU {gpu}; window {wid:06d} "
                        "will be requeued to the remaining decode workers."
                    )
                else:
                    eprint(result["traceback"])

                if decode_attempts[wid] > 2:
                    die(f"Decode window {wid} failed more than twice")
                if not active_decode_gpus:
                    die("All decode workers have failed")

                decoded_submitted.discard(wid)
                decode_task_q.put(result["task"])
                decoded_submitted.add(wid)
                decode_inflight += 1
            elif kind == "done":
                completed_files[wid] = Path(result["output"])
                decode_done += 1
                log(
                    f"[decode] {decode_done}/{len(jobs)} "
                    f"window={wid:06d} gpu={gpu} {result['seconds']:.1f}s"
                )
            else:
                die(f"Unexpected decode worker event: {kind}")

            # Emit every contiguous decoded window immediately.
            while next_assemble in completed_files:
                path = completed_files.pop(next_assemble)
                payload = torch_load_cpu(path)
                if payload.get("config_hash") != config_hash:
                    die(f"Decoded cache hash mismatch: {path}")
                job = jobs_by_id[next_assemble]
                assembler.consume(job, payload["frames"])
                del payload
                if not cfg.keep_decoded:
                    path.unlink(missing_ok=True)
                next_assemble += 1

            submit_ready_decodes()

        if next_assemble != len(jobs):
            die(
                f"Decode finished but assembler stopped at {next_assemble}/"
                f"{len(jobs)} windows"
            )

        writer.close()
    except BaseException:
        writer.terminate()
        raise
    finally:
        stop_workers(decode_task_q, decode_workers)
        if dit_workers and dit_task_q is not None:
            stop_workers(dit_task_q, dit_workers)

        # Explicitly close all parent-side queues and feeder threads. This is
        # important on very long runs: relying on multiprocessing's implicit
        # interpreter-shutdown cleanup can leave Python alive after "Completed".
        close_mp_queue(decode_task_q)
        if dit_task_q is not None:
            close_mp_queue(dit_task_q)
        close_mp_queue(result_q)

    expected = expected_output_frames(jobs, cfg.overlap_frames)
    if assembler.written != expected:
        die(
            f"Final video frame mismatch: wrote {assembler.written}, "
            f"expected {expected}"
        )
    log(f"Stage B+C wrote {assembler.written} frames.")


# ---------------------------------------------------------------------------
# Manifest / cache validation
# ---------------------------------------------------------------------------

def resolve_model_dir(cfg: Config) -> None:
    candidates = [
        Path(cfg.model_dir),
        Path(cfg.workspace) / "models",
        Path(cfg.workspace) / "vaeGGUF",
    ]
    for candidate in candidates:
        if (
            (candidate / cfg.dit_model).is_file()
            and (candidate / cfg.vae_model).is_file()
        ):
            cfg.model_dir = str(candidate.resolve())
            return
    die(
        "Could not find both FP16 model files. Searched: "
        + ", ".join(str(x) for x in candidates)
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_source_identity(path: Path) -> dict[str, Any]:
    """
    Build a fast, path-independent identity for a potentially huge source file.

    The absolute pathname and mtime are intentionally excluded so moving/copying
    an unchanged source does not invalidate its cache. For large files, hash
    five 4 MiB regions distributed across the file plus the exact byte size.
    Small files are hashed in full.
    """
    stat = path.stat()
    size = int(stat.st_size)
    chunk = SOURCE_FINGERPRINT_CHUNK_BYTES

    h = hashlib.sha256()
    h.update(f"seedvr2-source-identity-v{SOURCE_IDENTITY_VERSION}\0".encode())
    h.update(str(size).encode())
    h.update(b"\0")

    if size <= chunk * SOURCE_FINGERPRINT_SAMPLES:
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk)
                if not block:
                    break
                h.update(block)
        mode = "full"
        offsets = [0]
    else:
        max_offset = size - chunk
        offsets = sorted(
            {
                int(round(max_offset * i / (SOURCE_FINGERPRINT_SAMPLES - 1)))
                for i in range(SOURCE_FINGERPRINT_SAMPLES)
            }
        )
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                data = handle.read(chunk)
                if len(data) != chunk:
                    raise RuntimeError(
                        f"Short read while fingerprinting source at offset "
                        f"{offset}: {len(data)} of {chunk} bytes"
                    )
                h.update(offset.to_bytes(8, "little", signed=False))
                h.update(data)
        mode = "sampled"

    return {
        "version": SOURCE_IDENTITY_VERSION,
        "size": size,
        "mode": mode,
        "sample_bytes": chunk,
        "sample_count": len(offsets),
        "sampled_sha256": h.hexdigest(),
    }


def source_identity_matches(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    try:
        return (
            int(a["version"]) == int(b["version"])
            and int(a["size"]) == int(b["size"])
            and str(a["sampled_sha256"]) == str(b["sampled_sha256"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def build_portable_config_payload(
    cfg: Config,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    """
    Cache semantics that affect latent/output validity.

    Location-only values (input/output/cache paths and GPU assignments) are
    deliberately excluded. v13 is identified by file contents rather than path.
    """
    dit_path = Path(cfg.model_dir) / cfg.dit_model
    vae_path = Path(cfg.model_dir) / cfg.vae_model
    v13_path = Path(cfg.v13_runner)

    return {
        "cache_format": CACHE_FORMAT_VERSION,
        "source_identity_version": source_identity["version"],
        "source_size": source_identity["size"],
        "source_sampled_sha256": source_identity["sampled_sha256"],
        "dit_model": cfg.dit_model,
        "dit_size": dit_path.stat().st_size,
        "vae_model": cfg.vae_model,
        "vae_size": vae_path.stat().st_size,
        "window_frames": cfg.window_frames,
        "overlap_frames": cfg.overlap_frames,
        "resolution": cfg.resolution,
        "max_resolution": cfg.max_resolution,
        "encode_tiling": [cfg.encode_tile_size, cfg.encode_tile_overlap],
        "decode_tiling": [cfg.decode_tile_size, cfg.decode_tile_overlap],
        "attention": cfg.attention_mode,
        "color_correction": cfg.color_correction,
        "input_noise_scale": cfg.input_noise_scale,
        "latent_noise_scale": cfg.latent_noise_scale,
        "seed": cfg.seed,
        "scene_detect": cfg.scene_detect,
        "scene_threshold": cfg.scene_threshold,
        "test_start": cfg.test_start,
        "test_seconds": cfg.test_seconds,
        "postprocess_chunk_frames": cfg.postprocess_chunk_frames,
        "v13_runner_sha256": sha256_file(v13_path),
    }


def build_config_hash(
    cfg: Config,
    input_path: Path,
    source_identity: Optional[dict[str, Any]] = None,
) -> str:
    """
    Build the v24 portable cache hash.

    input_path is accepted for call-site compatibility but its pathname and
    mtime are not hashed.
    """
    if source_identity is None:
        source_identity = build_source_identity(input_path)
    payload = build_portable_config_payload(cfg, source_identity)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_legacy_v23_config_hash(
    cfg: Config,
    identity_path: Path,
    stat_path: Optional[Path] = None,
) -> str:
    """
    Reproduce the v23 path/mtime-sensitive hash for one-time cache migration.

    identity_path supplies the pathname embedded in the old hash. stat_path can
    be a moved copy when testing whether copy tools preserved size+mtime.
    """
    if stat_path is None:
        stat_path = identity_path
    stat = stat_path.stat()
    dit_path = Path(cfg.model_dir) / cfg.dit_model
    vae_path = Path(cfg.model_dir) / cfg.vae_model
    payload = {
        "cache_format": CACHE_FORMAT_VERSION,
        "input": str(identity_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "dit_model": cfg.dit_model,
        "dit_size": dit_path.stat().st_size,
        "vae_model": cfg.vae_model,
        "vae_size": vae_path.stat().st_size,
        "window_frames": cfg.window_frames,
        "overlap_frames": cfg.overlap_frames,
        "resolution": cfg.resolution,
        "max_resolution": cfg.max_resolution,
        "encode_tiling": [cfg.encode_tile_size, cfg.encode_tile_overlap],
        "decode_tiling": [cfg.decode_tile_size, cfg.decode_tile_overlap],
        "attention": cfg.attention_mode,
        "color_correction": cfg.color_correction,
        "input_noise_scale": cfg.input_noise_scale,
        "latent_noise_scale": cfg.latent_noise_scale,
        "seed": cfg.seed,
        "test_start": cfg.test_start,
        "test_seconds": cfg.test_seconds,
        "v13_runner": str(Path(cfg.v13_runner).resolve()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def manifest_config_semantically_matches(cfg: Config, old_cfg: Any) -> bool:
    """
    Compare cache-relevant settings while ignoring machine/location-only paths.
    """
    if not isinstance(old_cfg, dict):
        return False

    checks = {
        "dit_model": cfg.dit_model,
        "vae_model": cfg.vae_model,
        "window_frames": cfg.window_frames,
        "overlap_frames": cfg.overlap_frames,
        "resolution": cfg.resolution,
        "max_resolution": cfg.max_resolution,
        "encode_tile_size": cfg.encode_tile_size,
        "encode_tile_overlap": cfg.encode_tile_overlap,
        "decode_tile_size": cfg.decode_tile_size,
        "decode_tile_overlap": cfg.decode_tile_overlap,
        "attention_mode": cfg.attention_mode,
        "color_correction": cfg.color_correction,
        "input_noise_scale": cfg.input_noise_scale,
        "latent_noise_scale": cfg.latent_noise_scale,
        "seed": cfg.seed,
        "test_start": cfg.test_start,
        "test_seconds": cfg.test_seconds,
    }

    for key, current in checks.items():
        if key not in old_cfg:
            return False
        old = old_cfg[key]
        if isinstance(current, float):
            try:
                if abs(float(old) - current) > 1e-12:
                    return False
            except (TypeError, ValueError):
                return False
        elif old != current:
            return False

    # Scene settings were not part of the v23 hash but did affect window
    # planning, so require exact compatibility if present in the manifest.
    if bool(old_cfg.get("scene_detect")) != bool(cfg.scene_detect):
        return False
    try:
        if abs(float(old_cfg.get("scene_threshold")) - cfg.scene_threshold) > 1e-12:
            return False
    except (TypeError, ValueError):
        return False

    # Different postprocess chunk sizes do not invalidate encoded/restored
    # latents. If an old manifest already recorded the setting, require it only
    # when decoded cache files are intentionally being retained.
    if cfg.keep_decoded and "postprocess_chunk_frames" in old_cfg:
        try:
            if int(old_cfg["postprocess_chunk_frames"]) != int(
                cfg.postprocess_chunk_frames
            ):
                return False
        except (TypeError, ValueError):
            return False

    return True


def parse_manifest_plan(
    manifest: dict[str, Any],
) -> Optional[tuple[VideoInfo, list[WindowJob]]]:
    video_data = manifest.get("video")
    job_data = manifest.get("jobs")
    if not isinstance(video_data, dict) or not isinstance(job_data, list):
        return None

    try:
        info = VideoInfo(
            width=int(video_data["width"]),
            height=int(video_data["height"]),
            fps_num=int(video_data["fps_num"]),
            fps_den=int(video_data["fps_den"]),
            frames=int(video_data["frames"]),
            duration=float(video_data["duration"]),
        )
        jobs = [
            WindowJob(
                window_id=int(item["window_id"]),
                shot_id=int(item["shot_id"]),
                start_frame=int(item["start_frame"]),
                real_count=int(item["real_count"]),
                new_count=int(item["new_count"]),
                is_first=bool(item["is_first"]),
                is_last=bool(item["is_last"]),
                shot_frames=int(item["shot_frames"]),
            )
            for item in job_data
        ]
    except (KeyError, TypeError, ValueError):
        return None

    if not jobs or info.frames <= 0 or info.width <= 0 or info.height <= 0:
        return None
    if jobs[0].window_id != 0:
        return None
    if any(job.window_id != index for index, job in enumerate(jobs)):
        return None

    return info, jobs


def quick_probe_matches_manifest(path: Path, info: VideoInfo) -> bool:
    """
    Fast video identity sanity check that does NOT count every frame.
    """
    ffprobe = require_program("/usr/sbin/ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    if not data.get("streams"):
        return False

    stream = data["streams"][0]
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    rate = parse_rate(rate_text)
    duration = float(data.get("format", {}).get("duration") or 0.0)

    if int(stream["width"]) != info.width or int(stream["height"]) != info.height:
        return False
    if rate.numerator != info.fps_num or rate.denominator != info.fps_den:
        return False

    # Container duration can vary slightly after an otherwise lossless copy or
    # remux; allow roughly two source frames of tolerance.
    tolerance = max(0.100, 2.0 / max(float(rate), 1.0))
    return abs(duration - info.duration) <= tolerance


def load_matching_cache_plan(
    cache_root: Path,
    portable_config_hash: str,
    source_identity: dict[str, Any],
    cfg: Config,
    input_path: Path,
) -> Optional[tuple[VideoInfo, list[WindowJob], str, bool]]:
    """
    Reuse either a native v24 portable cache or adopt a v23 moved cache.

    Returns:
        (VideoInfo, jobs, cache_payload_hash, migrated_legacy_manifest)

    For a migrated v23 cache we deliberately KEEP its old config_hash as the
    payload hash. Existing encoded/restored .pt files therefore remain valid
    without rewriting hundreds of GiB. The manifest receives the new portable
    identity so future moves validate path-independently.
    """
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    plan = parse_manifest_plan(manifest)
    if plan is None:
        return None
    info, jobs = plan

    old_cfg = manifest.get("config")
    if not manifest_config_semantically_matches(cfg, old_cfg):
        return None

    cache_payload_hash = manifest.get("config_hash")
    if not isinstance(cache_payload_hash, str) or not cache_payload_hash:
        return None

    manifest_source = manifest.get("source_identity")
    manifest_portable_hash = manifest.get("portable_config_hash")

    # Native v24 (or already migrated) cache: strict portable validation.
    if manifest_source is not None or manifest_portable_hash is not None:
        if not source_identity_matches(manifest_source, source_identity):
            return None
        if manifest_portable_hash != portable_config_hash:
            return None
        return info, jobs, cache_payload_hash, False

    # --------------------- one-time v23 migration ---------------------
    old_input_text = old_cfg.get("input") if isinstance(old_cfg, dict) else None
    if not old_input_text:
        return None

    old_input_path = Path(str(old_input_text))
    verification = None

    # Strongest path: old source still exists. Compare sampled content directly.
    if old_input_path.is_file():
        old_identity = build_source_identity(old_input_path)
        if not source_identity_matches(old_identity, source_identity):
            return None
        verification = "sampled source content matches the original path"
    else:
        # If the copy preserved mtime and size, we can exactly reconstruct the
        # old v23 hash while substituting only the old pathname into its payload.
        try:
            reconstructed = build_legacy_v23_config_hash(
                cfg,
                identity_path=old_input_path,
                stat_path=input_path,
            )
        except OSError:
            reconstructed = None

        if reconstructed == cache_payload_hash:
            verification = "legacy v23 hash reconstructed from moved source metadata"
        else:
            # A copied source may have a new mtime, and v23 did not store an
            # independent content fingerprint. There is no cryptographic proof
            # available in that legacy manifest. Require same basename plus
            # exact stream geometry/rate and near-identical duration.
            if old_input_path.name != input_path.name:
                return None
            if not quick_probe_matches_manifest(input_path, info):
                return None
            verification = (
                "legacy fallback: same filename and matching video metadata "
                "(original source path is unavailable)"
            )

    log("Adopting moved legacy cache without rewriting cached latent files.")
    log(f"  Legacy verification: {verification}")
    log(f"  Preserving payload hash: {cache_payload_hash}")
    return info, jobs, cache_payload_hash, True



def prepare_cache(
    cfg: Config,
    info: VideoInfo,
    jobs: list[WindowJob],
    config_hash: str,
    portable_config_hash: str,
    source_identity: dict[str, Any],
    migrated_legacy_manifest: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    root = Path(cfg.cache_dir)
    if cfg.fresh_cache and root.exists():
        log(f"Removing old cache: {root}")
        shutil.rmtree(root)

    encoded = root / "encoded"
    restored = root / "restored"
    decoded = root / "decoded"
    logs_dir = root / "logs"
    for path in (root, encoded, restored, decoded, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    stale_parts = list(root.rglob("*.part-*"))
    if stale_parts:
        log(f"Removing {len(stale_parts)} stale atomic-save temp file(s).")
        for stale in stale_parts:
            stale.unlink(missing_ok=True)

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        old_hash = old.get("config_hash")
        if old_hash != config_hash:
            die(
                f"Existing cache manifest does not match this run.\n"
                f"  cache: {root}\n"
                f"  existing hash: {old_hash}\n"
                f"  current hash:  {config_hash}\n"
                f"Use --fresh-cache or a different --cache-dir."
            )

    manifest = {
        "script_version": SCRIPT_VERSION,
        # config_hash remains the hash embedded inside .pt cache payloads. A
        # migrated v23 cache intentionally retains its legacy value.
        "config_hash": config_hash,
        # portable_config_hash/source_identity are location-independent and are
        # used to validate future source/cache moves.
        "portable_config_hash": portable_config_hash,
        "source_identity": source_identity,
        "legacy_cache_adopted": bool(migrated_legacy_manifest),
        "created_or_validated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": asdict(cfg),
        "video": asdict(info),
        "window_count": len(jobs),
        "expected_output_frames": expected_output_frames(jobs, cfg.overlap_frames),
        "jobs": [asdict(job) for job in jobs],
    }
    atomic_json(manifest_path, manifest)

    return root, encoded, restored, decoded, logs_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def csv_gpu_list(value: str) -> list[str]:
    result = [x.strip() for x in value.split(",") if x.strip()]
    if not result:
        raise argparse.ArgumentTypeError("GPU list cannot be empty")
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SeedVR2 4-GPU phase-split FP16 upscaling runner"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input video; overrides INPUT_FILE in the script",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Final output; overrides OUTPUT_FILE in the script",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="RAM-backed latent cache; overrides CACHE_DIR in the script",
    )
    p.add_argument(
        "-d", "--directory",
        type=Path,
        default=None,
        help=(
            "Process every supported video in this directory sequentially. "
            "Each output is written next to the input as <filename>-AIU<ext>."
        ),
    )

    p.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    p.add_argument("--backend-dir", type=Path, default=DEFAULT_BACKEND_DIR)
    p.add_argument("--comfyui-dir", type=Path, default=DEFAULT_COMFYUI_DIR)
    p.add_argument("--v13-runner", type=Path, default=DEFAULT_V13_RUNNER)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--dit-model", default=DEFAULT_DIT_MODEL)
    p.add_argument("--vae-model", default=DEFAULT_VAE_MODEL)

    p.add_argument(
        "--encode-gpus",
        type=csv_gpu_list,
        default=list(DEFAULT_ENCODE_GPUS),
        help="Physical GPUs for pre-encode; default 3,0,1,4",
    )
    p.add_argument("--dit-gpu", default=DEFAULT_DIT_GPU)
    p.add_argument(
        "--decode-gpus",
        type=csv_gpu_list,
        default=list(DEFAULT_DECODE_GPUS),
        help="Physical GPUs for decode; default 0,1,4",
    )
    p.add_argument("--nvenc-gpu", default=DEFAULT_NVENC_GPU)

    p.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--overlap-frames", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    p.add_argument("--max-resolution", type=int, default=0)
    p.add_argument("--encode-tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--encode-tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP)
    p.add_argument("--decode-tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--decode-tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP)

    p.add_argument("--attention-mode", default="sdpa")
    p.add_argument(
        "--color-correction",
        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
        default="lab",
    )
    p.add_argument("--input-noise-scale", type=float, default=DEFAULT_INPUT_NOISE_SCALE)
    p.add_argument("--latent-noise-scale", type=float, default=DEFAULT_LATENT_NOISE_SCALE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument(
        "--scene-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--scene-threshold", type=float, default=0.32)
    p.add_argument("--test-start", type=float, default=600.0)
    p.add_argument(
        "--test-seconds",
        type=float,
        default=0.0,
        help="0 = full video; >0 processes only this test range",
    )

    p.add_argument("--nvenc-preset", default="p7")
    p.add_argument("--nvenc-cq", type=int, default=10)
    p.add_argument("--decode-inflight", type=int, default=DEFAULT_DECODE_INFLIGHT)
    p.add_argument(
        "--postprocess-chunk-frames",
        type=int,
        default=DEFAULT_POSTPROCESS_CHUNK_FRAMES,
        help="Temporal group size for memory-bounded Phase 4/LAB processing; default 17",
    )

    p.add_argument(
        "--keep-encoded",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--keep-restored",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--keep-decoded",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--fresh-cache", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Probe/scan and print the execution plan without launching GPU workers",
    )

    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        input=str(args.input.expanduser().resolve()),
        output=str(args.output.expanduser().resolve()),
        cache_dir=str(args.cache_dir.expanduser().resolve()),
        workspace=str(args.workspace.expanduser().resolve()),
        backend_dir=str(args.backend_dir.expanduser().resolve()),
        comfyui_dir=str(args.comfyui_dir.expanduser().resolve()),
        v13_runner=str(args.v13_runner.expanduser().resolve()),
        model_dir=str(args.model_dir.expanduser().resolve()),
        dit_model=args.dit_model,
        vae_model=args.vae_model,
        encode_gpus=list(args.encode_gpus),
        dit_gpu=str(args.dit_gpu),
        decode_gpus=list(args.decode_gpus),
        nvenc_gpu=str(args.nvenc_gpu),
        window_frames=args.window_frames,
        overlap_frames=args.overlap_frames,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        encode_tile_size=args.encode_tile_size,
        encode_tile_overlap=args.encode_tile_overlap,
        decode_tile_size=args.decode_tile_size,
        decode_tile_overlap=args.decode_tile_overlap,
        attention_mode=args.attention_mode,
        color_correction=args.color_correction,
        input_noise_scale=args.input_noise_scale,
        latent_noise_scale=args.latent_noise_scale,
        seed=args.seed,
        scene_detect=bool(args.scene_detect),
        scene_threshold=args.scene_threshold,
        test_start=args.test_start,
        test_seconds=args.test_seconds,
        nvenc_preset=args.nvenc_preset,
        nvenc_cq=args.nvenc_cq,
        decode_inflight=args.decode_inflight,
        postprocess_chunk_frames=args.postprocess_chunk_frames,
        keep_encoded=bool(args.keep_encoded),
        keep_restored=bool(args.keep_restored),
        keep_decoded=bool(args.keep_decoded),
        fresh_cache=bool(args.fresh_cache),
        debug=bool(args.debug),
        dit_min_free_mib=DEFAULT_DIT_MIN_FREE_MIB,
        encode_min_free_mib=dict(DEFAULT_ENCODE_MIN_FREE_MIB),
        decode_min_free_mib=dict(DEFAULT_DECODE_MIN_FREE_MIB),
    )


def validate_config(cfg: Config) -> None:
    input_path = Path(cfg.input)
    if not input_path.is_file():
        die(f"Input file not found: {input_path}")
    if not Path(cfg.backend_dir, "inference_cli.py").is_file():
        die(f"SeedVR2 backend not found: {cfg.backend_dir}")
    if not Path(cfg.v13_runner).is_file():
        die(f"v13 compatibility runner not found: {cfg.v13_runner}")
    if not Path(cfg.comfyui_dir).is_dir():
        die(f"ComfyUI checkout not found: {cfg.comfyui_dir}")

    if not valid_seedvr2_batch(cfg.window_frames):
        die(
            f"--window-frames must satisfy 4n+1 and be >=5; "
            f"got {cfg.window_frames}"
        )
    if not (0 <= cfg.overlap_frames < cfg.window_frames):
        die("--overlap-frames must be >=0 and < window size")
    if cfg.decode_inflight < len(cfg.decode_gpus):
        die("--decode-inflight must be at least the number of decode GPUs")
    if cfg.postprocess_chunk_frames < 1:
        die("--postprocess-chunk-frames must be >= 1")
    if cfg.postprocess_chunk_frames > cfg.window_frames:
        die("--postprocess-chunk-frames cannot exceed --window-frames")

    if "2" in cfg.encode_gpus or "2" in cfg.decode_gpus or cfg.dit_gpu == "2":
        die(
            "CUDA device 2 is reserved as the system display GPU in this "
            "configuration and must not be used"
        )

    resolve_model_dir(cfg)


def print_plan(
    cfg: Config,
    info: VideoInfo,
    jobs: list[WindowJob],
    cache_root: Path,
) -> None:
    source_frames = expected_output_frames(jobs, cfg.overlap_frames)
    raw_window_frames = sum(cfg.window_frames for _ in jobs)
    overlap_extra = raw_window_frames - source_frames

    log("\nSeedVR2 4-GPU phase-split plan")
    log(f"  Version:          {SCRIPT_VERSION}")
    log(f"  Input:            {cfg.input}")
    log(f"  Output:           {cfg.output}")
    log(f"  Cache:            {cache_root}")
    log(f"  Source:           {info.width}x{info.height} @ {info.fps_text}")
    log(f"  Source frames:    {source_frames} (FFprobe-counted readable frames)")
    log(f"  Window:           {cfg.window_frames}")
    log(f"  Overlap:          {cfg.overlap_frames}")
    log(f"  Windows:          {len(jobs)}")
    log(f"  Computed frames:  {raw_window_frames}")
    log(f"  Overlap/padding:  {overlap_extra} frame-instances")
    log(f"  Stage A encode:   GPUs {','.join(cfg.encode_gpus)}")
    log(f"  Stage B DiT:      GPU {cfg.dit_gpu}")
    log(f"  Stage C decode:   GPUs {','.join(cfg.decode_gpus)}")
    if cfg.nvenc_gpu in cfg.decode_gpus:
        log(
            f"  Output NVENC:     GPU {cfg.nvenc_gpu} "
            "(shares a Stage C decode GPU)"
        )
    elif cfg.nvenc_gpu == cfg.dit_gpu:
        log(
            f"  Output NVENC:     GPU {cfg.nvenc_gpu} "
            "(shares the Stage B DiT GPU)"
        )
    else:
        log(f"  Output NVENC:     GPU {cfg.nvenc_gpu}")
    log(
        f"  VAE encode tile:  {cfg.encode_tile_size}/"
        f"{cfg.encode_tile_overlap}"
    )
    log(
        f"  VAE decode tile:  {cfg.decode_tile_size}/"
        f"{cfg.decode_tile_overlap}"
    )
    log(f"  Phase 4 chunks:   {cfg.postprocess_chunk_frames} frames (LAB/normalization only)")
    log(f"  DiT:              {Path(cfg.model_dir) / cfg.dit_model}")
    log(f"  VAE:              {Path(cfg.model_dir) / cfg.vae_model}")
    log(f"  Attention:        {cfg.attention_mode}")
    log(f"  Color correction: {cfg.color_correction}")
    log(f"  RAM-disk status:  {cache_disk_status(cache_root)}")


# ---------------------------------------------------------------------------
# Directory-mode helpers
# ---------------------------------------------------------------------------

def is_supported_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def derive_directory_mode_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}-AIU{path.suffix}")


def derive_directory_mode_cache_dir(cache_root: Path, path: Path) -> Path:
    safe_suffix = path.suffix.lower().lstrip(".") or "video"
    return cache_root / f"{path.stem}-{safe_suffix}-phase-cache"


def list_directory_mode_inputs(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not is_supported_video_file(path):
            continue
        if path.stem.endswith("-AIU"):
            continue
        candidates.append(path)
    return candidates


def run_pipeline_for_config(cfg: Config, args: argparse.Namespace) -> int:
    input_path = Path(cfg.input)
    output_path = Path(cfg.output)
    cache_root = Path(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Build a fast path-independent identity before inspecting the manifest.
    # Moving/copying an unchanged source no longer changes cache validity.
    log("Fingerprinting source for portable cache identity...")
    source_identity = build_source_identity(input_path)
    portable_config_hash = build_config_hash(
        cfg,
        input_path,
        source_identity=source_identity,
    )

    cached_plan = None
    cache_payload_hash = portable_config_hash
    migrated_legacy_manifest = False

    if not cfg.fresh_cache:
        cached_plan = load_matching_cache_plan(
            cache_root,
            portable_config_hash,
            source_identity,
            cfg,
            input_path,
        )

    if cached_plan is not None:
        info, jobs, cache_payload_hash, migrated_legacy_manifest = cached_plan
        log(
            f"Reusing cached source/window plan: {info.frames} frames, "
            f"{len(jobs)} windows."
        )
        log("Skipping FFprobe frame count and scene-detection scans.")
    else:
        manifest_path = cache_root / "manifest.json"
        if cfg.fresh_cache:
            log("Fresh cache requested; rebuilding source/window plan.")
        elif manifest_path.is_file():
            log(
                "Cache manifest is not reusable for this source/configuration; "
                "rebuilding source/window plan."
            )
        else:
            log("No reusable cache manifest found; building source/window plan.")

        info = probe_video(input_path)

        if cfg.test_seconds > 0:
            boundaries = [0, info.frames]
            if cfg.scene_detect:
                log(
                    "Test-range mode: hard-cut scene detection is disabled "
                    "for the test subset."
                )
        elif cfg.scene_detect:
            log("Scanning source for hard scene cuts...")
            boundaries = detect_scene_boundaries(
                input_path,
                info,
                cfg.scene_threshold,
            )
            log(f"Detected {max(0, len(boundaries)-2)} hard cuts.")
        else:
            boundaries = [0, info.frames]

        jobs = build_jobs(
            info,
            boundaries=boundaries,
            window=cfg.window_frames,
            overlap=cfg.overlap_frames,
            test_start=cfg.test_start,
            test_seconds=cfg.test_seconds,
        )
        if not jobs:
            die("No window jobs were generated")

    root, encoded_dir, restored_dir, decoded_dir, logs_dir = prepare_cache(
        cfg,
        info,
        jobs,
        cache_payload_hash,
        portable_config_hash,
        source_identity,
        migrated_legacy_manifest=migrated_legacy_manifest,
    )

    print_plan(cfg, info, jobs, root)
    log(f"  Cache payload:    {cache_payload_hash}")
    log(f"  Portable hash:    {portable_config_hash}")

    if args.plan_only:
        return 0

    started = time.monotonic()

    stage_encode(cfg, info, jobs, encoded_dir, cache_payload_hash)
    log(f"Stage A complete. Cache status: {cache_disk_status(root)}")

    final_video_only = root / "final.video-only.mkv"
    stage_dit_decode_concurrent(
        cfg,
        info,
        jobs,
        encoded_dir,
        restored_dir,
        decoded_dir,
        final_video_only,
        cache_payload_hash,
    )
    log(f"Stage B+C complete. Cache status: {cache_disk_status(root)}")

    if cfg.test_seconds > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_video_only, output_path)
        log("Test-range mode: output is video-only; source streams are not remuxed.")
    else:
        remux_original_streams(final_video_only, input_path, output_path)

    if not cfg.keep_encoded:
        shutil.rmtree(encoded_dir, ignore_errors=True)
    if not cfg.keep_restored:
        shutil.rmtree(restored_dir, ignore_errors=True)
    if not cfg.keep_decoded:
        shutil.rmtree(decoded_dir, ignore_errors=True)

    elapsed = time.monotonic() - started
    log("\nCompleted.")
    log(f"  Output:        {output_path}")
    log(f"  Cache:         {root}")
    log(f"  Wall seconds:  {elapsed:.0f}")
    log(f"  Wall hours:    {elapsed/3600:.2f}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _main_impl(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.directory is not None:
        directory = args.directory.expanduser().resolve()
        if not directory.is_dir():
            die(f"Directory not found: {directory}")

        inputs = list_directory_mode_inputs(directory)
        if not inputs:
            die(
                f"No supported video files found in {directory}. "
                f"Extensions: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
            )

        cache_parent = args.cache_dir.expanduser().resolve()
        cache_parent.mkdir(parents=True, exist_ok=True)

        log("Directory mode enabled.")
        log(f"  Directory:      {directory}")
        log(f"  Files queued:   {len(inputs)}")
        log(f"  Cache parent:   {cache_parent}")

        failures: list[tuple[Path, str]] = []
        for idx, input_path in enumerate(inputs, start=1):
            output_path = derive_directory_mode_output(input_path)
            per_file_cache = derive_directory_mode_cache_dir(cache_parent, input_path)

            log("\n" + "=" * 78)
            log(
                f"Directory job {idx}/{len(inputs)}\n"
                f"  Input:   {input_path}\n"
                f"  Output:  {output_path}\n"
                f"  Cache:   {per_file_cache}"
            )
            log("=" * 78)

            per_file_args = argparse.Namespace(**vars(args))
            per_file_args.input = input_path
            per_file_args.output = output_path
            per_file_args.cache_dir = per_file_cache
            per_file_args.directory = None

            cfg = config_from_args(per_file_args)
            validate_config(cfg)

            try:
                run_pipeline_for_config(cfg, per_file_args)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                failures.append((input_path, f"SystemExit({code})"))
                log(f"FAILED: {input_path} -> exit status {code}")
            except Exception as exc:  # pragma: no cover - defensive top-level guard
                failures.append((input_path, f"{type(exc).__name__}: {exc}"))
                log(f"FAILED: {input_path} -> {type(exc).__name__}: {exc}")

        log("\nDirectory mode complete.")
        log(f"  Total files:    {len(inputs)}")
        log(f"  Successful:     {len(inputs) - len(failures)}")
        log(f"  Failed:         {len(failures)}")
        if failures:
            log("  Failure summary:")
            for path, reason in failures:
                log(f"    - {path}: {reason}")
            return 1
        return 0

    cfg = config_from_args(args)
    validate_config(cfg)
    return run_pipeline_for_config(cfg, args)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Top-level shell-safety wrapper.

    Capture the terminal before any worker/subprocess starts, then guarantee
    that no multiprocessing children or altered TTY flags survive the run.
    """
    tty_state = capture_terminal_state()
    try:
        return _main_impl(argv)
    finally:
        cleanup_child_processes()
        restore_terminal_state(tty_state)


if __name__ == "__main__":
    # Never fork a process that may later initialize CUDA.
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
