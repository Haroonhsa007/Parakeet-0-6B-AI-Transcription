#!/usr/bin/env python3
"""
Parakeet MLX benchmark runner for Apple Silicon.

Features:
- Uses a project-local model cache and downloads missing model files automatically.
- Loads the INT8 encoder / BF16 decoder checkpoint.
- Forces MLX weight evaluation before recording load time.
- Separates download, model load, warm-up, and inference measurements.
- Runs repeat benchmarks and reports median / min / max / p95.
- Captures process RSS, system memory, swap, CPU, and MLX Metal memory.
- Optional duration-scaling tests and JSON/CSV-friendly results.

Install:
  pip install parakeet-mlx huggingface_hub soundfile psutil numpy

Run:
  python transcribe.py --audio audio.wav
  python transcribe.py --audio audio.wav --repeats 5 --warmups 1
  python transcribe.py --audio audio.wav --duration-tests 30,60,120
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import psutil
import soundfile as sf
from huggingface_hub import hf_hub_download
from parakeet_mlx.utils import from_config


DEFAULT_MODEL_ID = "sonic-speech/parakeet-tdt-0.6b-v3-int8"
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / DEFAULT_MODEL_ID.replace("/", "--")
RESULTS_DIR = PROJECT_DIR / "benchmark_results"
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "quantization_config.json",
)


def mb(n: int | float) -> float:
    return round(float(n) / (1024 ** 2), 2)


def run_command(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT, timeout=5
        ).strip()
    except Exception:
        return "Unavailable"


def sysctl_value(key: str) -> str:
    return run_command(["sysctl", "-n", key])


def swap_info() -> dict[str, Any]:
    raw = run_command(["sysctl", "vm.swapusage"])
    return {"raw": raw}


def memory_pressure() -> str:
    return run_command(["memory_pressure"])


def system_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "ram_total_mb": mb(vm.total),
        "ram_used_mb": mb(vm.used),
        "ram_available_mb": mb(vm.available),
        "ram_percent": vm.percent,
        "swap": swap_info(),
        "memory_pressure": memory_pressure(),
        "cpu_percent_system": psutil.cpu_percent(interval=0.2),
    }


class ResourceMonitor:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.samples: list[dict[str, float]] = []
        self.running = False
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            self.samples.append({
                "time": time.perf_counter(),
                "rss_mb": mb(self.proc.memory_info().rss),
                "cpu_percent": self.proc.cpu_percent(interval=None),
                "system_ram_used_mb": mb(psutil.virtual_memory().used),
                "system_ram_available_mb": mb(psutil.virtual_memory().available),
            })
        except (psutil.Error, OSError):
            pass

    def _loop(self) -> None:
        while self.running:
            self._sample()
            time.sleep(self.interval)

    def start(self) -> None:
        self.samples.clear()
        self._sample()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.running = False
        if self.thread:
            self.thread.join()
        self._sample()
        if not self.samples:
            return {}
        rss = [x["rss_mb"] for x in self.samples]
        cpu = [x["cpu_percent"] for x in self.samples]
        sys_used = [x["system_ram_used_mb"] for x in self.samples]
        return {
            "sample_count": len(self.samples),
            "peak_process_rss_mb": max(rss),
            "average_process_rss_mb": round(statistics.mean(rss), 2),
            "peak_process_cpu_percent": round(max(cpu), 2),
            "average_process_cpu_percent": round(statistics.mean(cpu), 2),
            "peak_system_ram_used_mb": max(sys_used),
            "minimum_system_ram_available_mb": min(
                x["system_ram_available_mb"] for x in self.samples
            ),
        }


def metal_snapshot() -> dict[str, Any]:
    try:
        return {
            "active_mb": mb(mx.get_active_memory()),
            "peak_mb": mb(mx.get_peak_memory()),
            "cache_mb": mb(mx.get_cache_memory()),
        }
    except Exception as exc:
        return {"error": str(exc)}


def audio_metadata(audio_path: Path) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    info = sf.info(str(audio_path))
    return {
        "path": str(audio_path.resolve()),
        "size_mb": round(audio_path.stat().st_size / (1024 ** 2), 3),
        "duration_seconds": round(info.duration, 3),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }


def ensure_model(model_id: str, model_dir: Path) -> dict[str, Any]:
    """Use complete project-local files when present; download missing files."""
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    downloaded: list[str] = []
    paths: dict[str, str] = {}

    for filename in REQUIRED_MODEL_FILES:
        local_path = model_dir / filename
        if not local_path.is_file() or local_path.stat().st_size == 0:
            logging.info("Missing model file: %s. Downloading into %s",
                         filename, model_dir)
            result = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(model_dir),
            )
            local_path = Path(result)
            downloaded.append(filename)
        else:
            logging.info("Using project-local model file: %s", local_path)
        paths[filename] = str(local_path.resolve())

    return {
        "model_id": model_id,
        "model_directory": str(model_dir.resolve()),
        "downloaded_files": downloaded,
        "file_paths": paths,
        "download_or_cache_check_seconds": round(time.perf_counter() - started, 3),
        "file_sizes_mb": {
            name: round(Path(path).stat().st_size / (1024 ** 2), 2)
            for name, path in paths.items()
        },
    }


def load_model(file_paths: dict[str, str]) -> tuple[Any, dict[str, Any]]:
    with open(file_paths["config.json"], encoding="utf-8") as f:
        config = json.load(f)
    with open(file_paths["quantization_config.json"], encoding="utf-8") as f:
        qconfig = json.load(f)

    model = from_config(config)
    bits = int(qconfig["bits"])
    group_size = int(qconfig["group_size"])

    nn.quantize(
        model,
        bits=bits,
        group_size=group_size,
        class_predicate=lambda path, layer: (
            path.startswith("encoder.") and isinstance(layer, nn.Linear)
        ),
    )
    model.load_weights(file_paths["model.safetensors"])

    # MLX evaluates lazily. Force parameter arrays to materialize before
    # stopping the model-load timer. This is a more meaningful load measure.
    mx.eval(model.parameters())
    mx.synchronize()

    return model, {
        "bits": bits,
        "group_size": group_size,
        "quantization_config": qconfig,
        "weights_forced_evaluated": True,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 4)


def transcribe_once(model: Any, audio_path: Path) -> tuple[Any, float]:
    start = time.perf_counter()
    result = model.transcribe(str(audio_path))
    mx.synchronize()
    elapsed = time.perf_counter() - start
    return result, elapsed


def make_duration_test_files(
    source: Path, durations: list[int], output_dir: Path
) -> list[tuple[int, Path]]:
    audio, sr = sf.read(str(source), dtype="float32", always_2d=True)
    total = len(audio) / sr
    generated: list[tuple[int, Path]] = []
    for seconds in durations:
        if seconds <= 0 or seconds > total:
            logging.warning("Skipping %ss test, source audio is %.1fs",
                            seconds, total)
            continue
        target = output_dir / f"duration_test_{seconds}s.wav"
        sf.write(str(target), audio[: int(seconds * sr)], sr, subtype="PCM_16")
        generated.append((seconds, target))
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="Parakeet MLX benchmark")
    parser.add_argument("--audio", type=Path, default=PROJECT_DIR / "audio.wav")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--duration-tests",
        default="",
        help="Comma-separated seconds, e.g. 30,60,120. Uses prefixes of input audio.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.2)
    args = parser.parse_args()

    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be >= 1 and --warmups must be >= 0")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"benchmark_{run_id}.log"
    json_path = RESULTS_DIR / f"benchmark_{run_id}.json"
    transcript_path = RESULTS_DIR / f"transcript_{run_id}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_model": sysctl_value("hw.model"),
            "cpu_brand": sysctl_value("machdep.cpu.brand_string"),
            "logical_cpu_count": psutil.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "parakeet_mlx_version": "not queried",
        },
        "settings": {
            "repeats": args.repeats,
            "warmups": args.warmups,
            "sample_interval_seconds": args.sample_interval,
        },
        "system_before": system_snapshot(),
    }

    try:
        report["audio"] = audio_metadata(args.audio.resolve())

        logging.info("Checking model files in %s", args.model_dir)
        model_info = ensure_model(args.model_id, args.model_dir)
        report["model_files"] = model_info

        monitor = ResourceMonitor(args.sample_interval)
        logging.info("Loading model and materializing MLX weights")
        monitor.start()
        load_start = time.perf_counter()
        model, quant_info = load_model(model_info["file_paths"])
        model_load_seconds = time.perf_counter() - load_start
        load_resources = monitor.stop()

        report["quantization"] = quant_info
        report["model_load"] = {
            "seconds": round(model_load_seconds, 4),
            "resources": load_resources,
            "metal_after_load": metal_snapshot(),
            "system_after_load": system_snapshot(),
        }
        logging.info("Model load: %.3fs", model_load_seconds)

        # Warm-up is measured separately and excluded from repeat statistics.
        warmup_times: list[float] = []
        for idx in range(args.warmups):
            logging.info("Warm-up %d/%d", idx + 1, args.warmups)
            monitor.start()
            _, elapsed = transcribe_once(model, args.audio.resolve())
            warmup_times.append(elapsed)
            warmup_resources = monitor.stop()
            logging.info("Warm-up time %.3fs, resources %s",
                         elapsed, warmup_resources)

        report["warmup"] = {
            "runs_seconds": [round(x, 4) for x in warmup_times],
            "excluded_from_repeat_statistics": True,
        }

        repeat_times: list[float] = []
        repeat_resources: list[dict[str, Any]] = []
        final_result = None
        for idx in range(args.repeats):
            try:
                mx.reset_peak_memory()
            except Exception:
                pass
            monitor.start()
            result, elapsed = transcribe_once(model, args.audio.resolve())
            resources = monitor.stop()
            repeat_times.append(elapsed)
            repeat_resources.append(resources)
            final_result = result
            logging.info(
                "Run %d/%d: %.3fs, RTF %.5f, RTFx %.2fx",
                idx + 1, args.repeats, elapsed,
                elapsed / report["audio"]["duration_seconds"],
                report["audio"]["duration_seconds"] / elapsed,
            )

        duration = report["audio"]["duration_seconds"]
        words = len(final_result.text.split()) if final_result else 0
        report["inference"] = {
            "runs_seconds": [round(x, 4) for x in repeat_times],
            "min_seconds": round(min(repeat_times), 4),
            "max_seconds": round(max(repeat_times), 4),
            "mean_seconds": round(statistics.mean(repeat_times), 4),
            "median_seconds": round(statistics.median(repeat_times), 4),
            "p95_seconds": percentile(repeat_times, 0.95),
            "standard_deviation_seconds": (
                round(statistics.stdev(repeat_times), 4)
                if len(repeat_times) > 1 else 0.0
            ),
            "audio_seconds": duration,
            "rtf_median": round(statistics.median(repeat_times) / duration, 5),
            "rtfx_median": round(duration / statistics.median(repeat_times), 2),
            "words": words,
            "words_per_second_median": round(
                words / statistics.median(repeat_times), 2
            ),
            "sentence_count": len(final_result.sentences) if final_result else 0,
            "resources_per_run": repeat_resources,
            "metal_after_inference": metal_snapshot(),
        }

        # Duration scaling benchmark. These are prefixes of the source file.
        requested_durations = [
            int(x.strip()) for x in args.duration_tests.split(",") if x.strip()
        ]
        duration_results = []
        if requested_durations:
            temp_dir = RESULTS_DIR / f"duration_inputs_{run_id}"
            temp_dir.mkdir(exist_ok=True)
            for seconds, test_path in make_duration_test_files(
                args.audio.resolve(), requested_durations, temp_dir
            ):
                times = []
                for _ in range(args.repeats):
                    _, elapsed = transcribe_once(model, test_path)
                    times.append(elapsed)
                median_time = statistics.median(times)
                duration_results.append({
                    "audio_seconds": seconds,
                    "runs_seconds": [round(x, 4) for x in times],
                    "median_seconds": round(median_time, 4),
                    "rtf_median": round(median_time / seconds, 5),
                    "rtfx_median": round(seconds / median_time, 2),
                })
            report["duration_scaling_tests"] = duration_results

        if final_result:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(final_result.text + "\n\n")
                for sentence in final_result.sentences:
                    f.write(
                        f"[{sentence.start:.3f}s - {sentence.end:.3f}s] "
                        f"{sentence.text}\n"
                    )

        report["system_after"] = system_snapshot()
        report["outputs"] = {
            "transcript": str(transcript_path),
            "json": str(json_path),
            "log": str(log_path),
        }
        report["total_elapsed_seconds"] = round(
            model_info["download_or_cache_check_seconds"]
            + model_load_seconds
            + sum(warmup_times)
            + sum(repeat_times),
            3,
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logging.info("Median inference: %.3fs", report["inference"]["median_seconds"])
        logging.info("Median RTFx: %.2fx", report["inference"]["rtfx_median"])
        logging.info("Peak process RSS across repeat runs: %.2f MB",
                     max((x.get("peak_process_rss_mb", 0)
                          for x in repeat_resources), default=0))
        logging.info("Results JSON: %s", json_path)
        logging.info("Transcript: %s", transcript_path)
        logging.info("Benchmark complete")
        print("\nTRANSCRIPT\n")
        print(final_result.text if final_result else "")
        return 0

    except Exception as exc:
        report["failure"] = {
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
