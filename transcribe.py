
#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import platform
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone

import psutil
import soundfile as sf
import mlx.core as mx
import mlx.nn as nn

from huggingface_hub import hf_hub_download
from parakeet_mlx.utils import from_config


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "sonic-speech/parakeet-tdt-0.6b-v3-int8"
AUDIO_FILE = Path("audio.wav")

OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_INTERVAL = 0.25

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = OUTPUT_DIR / f"benchmark_{RUN_ID}.log"
JSON_FILE = OUTPUT_DIR / f"benchmark_{RUN_ID}.json"
TXT_FILE = OUTPUT_DIR / f"transcript_{RUN_ID}.txt"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)

log = logging.getLogger("PARAKEET")


def section(title):
    log.info("=" * 65)
    log.info(title)
    log.info("=" * 65)


def mb(value):
    return round(value / (1024 ** 2), 2)


def get_sysctl(name):
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name],
            text=True
        ).strip()
    except Exception:
        return "Unavailable"


# ============================================================
# MAC HARDWARE INFORMATION
# ============================================================

def get_hardware_info():

    vm = psutil.virtual_memory()

    return {
        "machine": platform.machine(),
        "mac_model": get_sysctl("hw.model"),
        "chip": get_sysctl("machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "mlx_version": getattr(mx, "__version__", "Unknown"),
    }


# ============================================================
# RESOURCE MONITOR
# ============================================================

class ResourceMonitor:

    def __init__(self):

        self.process = psutil.Process(os.getpid())

        self.running = False
        self.thread = None

        self.peak_rss = 0
        self.peak_system_ram_used = 0

        self.cpu_samples = []
        self.rss_samples = []

    def sample(self):

        try:
            cpu = self.process.cpu_percent(interval=None)
            rss = self.process.memory_info().rss
            system_ram = psutil.virtual_memory().used

            self.cpu_samples.append(cpu)
            self.rss_samples.append(mb(rss))

            self.peak_rss = max(self.peak_rss, rss)
            self.peak_system_ram_used = max(
                self.peak_system_ram_used,
                system_ram
            )

        except (psutil.Error, OSError):
            pass

    def loop(self):

        while self.running:
            self.sample()
            time.sleep(SAMPLE_INTERVAL)

    def start(self):

        self.sample()

        self.running = True

        self.thread = threading.Thread(
            target=self.loop,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join()

        self.sample()

    def report(self):

        return {
            "peak_process_ram_mb": mb(self.peak_rss),
            "average_process_ram_mb": (
                round(sum(self.rss_samples) / len(self.rss_samples), 2)
                if self.rss_samples else 0
            ),
            "average_cpu_percent": (
                round(sum(self.cpu_samples) / len(self.cpu_samples), 2)
                if self.cpu_samples else 0
            ),
            "peak_cpu_percent": (
                round(max(self.cpu_samples), 2)
                if self.cpu_samples else 0
            ),
            "peak_system_ram_used_mb": mb(
                self.peak_system_ram_used
            )
        }


# ============================================================
# MLX / METAL MEMORY
# ============================================================

def metal_memory():

    try:
        return {
            "active_mb": mb(mx.get_active_memory()),
            "peak_mb": mb(mx.get_peak_memory()),
            "cache_mb": mb(mx.get_cache_memory())
        }

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# AUDIO INFORMATION
# ============================================================

def inspect_audio(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path.resolve()}"
        )

    info = sf.info(str(path))

    duration = info.duration

    return {
        "file": str(path.resolve()),
        "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
        "duration_seconds": round(duration, 3),
        "duration_minutes": round(duration / 60, 3),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype
    }


# ============================================================
# BENCHMARK
# ============================================================

def main():

    benchmark = {
        "run_id": RUN_ID,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": MODEL_ID,
    }

    monitor = ResourceMonitor()

    # --------------------------------------------------------
    # SYSTEM
    # --------------------------------------------------------

    section("MAC MINI HARDWARE")

    hardware = get_hardware_info()

    benchmark["hardware"] = hardware

    for key, value in hardware.items():
        log.info("%s: %s", key, value)

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    section("AUDIO INSPECTION")

    audio = inspect_audio(AUDIO_FILE)

    benchmark["audio"] = audio

    for key, value in audio.items():
        log.info("%s: %s", key, value)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_ram = psutil.virtual_memory()

    benchmark["system_baseline"] = {
        "ram_used_mb": mb(baseline_ram.used),
        "ram_available_mb": mb(baseline_ram.available),
        "ram_percent": baseline_ram.percent,
        "cpu_percent": psutil.cpu_percent(interval=1)
    }

    log.info(
        "Available RAM: %.2f MB",
        mb(baseline_ram.available)
    )

    # --------------------------------------------------------
    # DOWNLOAD / CACHE
    # --------------------------------------------------------

    section("MODEL DOWNLOAD / CACHE")

    download_start = time.perf_counter()

    config_path = hf_hub_download(
        MODEL_ID,
        "config.json"
    )

    weights_path = hf_hub_download(
        MODEL_ID,
        "model.safetensors"
    )

    quant_config_path = hf_hub_download(
        MODEL_ID,
        "quantization_config.json"
    )

    download_time = time.perf_counter() - download_start

    benchmark["model_download_seconds"] = round(
        download_time, 3
    )

    log.info("Model files ready in %.3f seconds", download_time)

    # --------------------------------------------------------
    # MODEL INITIALIZATION
    # --------------------------------------------------------

    section("MODEL INITIALIZATION")

    monitor.start()

    load_start = time.perf_counter()

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    with open(quant_config_path, encoding="utf-8") as f:
        quant_config = json.load(f)

    model = from_config(config)

    log.info(
        "Quantization: %s",
        json.dumps(quant_config)
    )

    bits = quant_config["bits"]
    group_size = quant_config["group_size"]

    # Quantize encoder linear layers only.
    # Decoder and joint network remain BF16.

    nn.quantize(
        model,
        bits=bits,
        group_size=group_size,
        class_predicate=lambda path, layer: (
            path.startswith("encoder.")
            and isinstance(layer, nn.Linear)
        )
    )

    model.load_weights(weights_path)

    load_time = time.perf_counter() - load_start

    benchmark["model_initialization_seconds"] = round(
        load_time, 3
    )

    log.info("Model initialization: %.3f seconds", load_time)

    benchmark["metal_after_model_load"] = metal_memory()

    log.info(
        "MLX memory: %s",
        benchmark["metal_after_model_load"]
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    section("TRANSCRIPTION BENCHMARK")

    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    inference_start = time.perf_counter()

    result = model.transcribe(str(AUDIO_FILE))

    inference_time = time.perf_counter() - inference_start

    monitor.stop()

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    duration = audio["duration_seconds"]

    rtf = inference_time / duration if duration else 0

    rtf_x = duration / inference_time if inference_time else 0

    words = len(result.text.split())

    benchmark["inference"] = {
        "inference_seconds": round(inference_time, 4),
        "audio_seconds": duration,
        "rtf": round(rtf, 5),
        "rtfx": round(rtf_x, 2),
        "audio_seconds_per_second": round(rtf_x, 2),
        "transcribed_words": words,
        "words_per_second": round(
            words / inference_time, 2
        ) if inference_time else 0,
        "sentence_count": len(result.sentences),
    }

    benchmark["process_resources"] = monitor.report()

    benchmark["metal_after_inference"] = metal_memory()

    # --------------------------------------------------------
    # SYSTEM AFTER INFERENCE
    # --------------------------------------------------------

    vm = psutil.virtual_memory()

    benchmark["system_after_inference"] = {
        "ram_used_mb": mb(vm.used),
        "ram_available_mb": mb(vm.available),
        "ram_percent": vm.percent,
        "cpu_percent": psutil.cpu_percent(interval=1)
    }

    # --------------------------------------------------------
    # TRANSCRIPT
    # --------------------------------------------------------

    section("TRANSCRIPTION OUTPUT")

    print("\n" + result.text + "\n")

    with open(TXT_FILE, "w", encoding="utf-8") as f:

        f.write(result.text + "\n\n")

        for sentence in result.sentences:

            f.write(
                f"[{sentence.start:.3f}s - "
                f"{sentence.end:.3f}s] "
                f"{sentence.text}\n"
            )

    benchmark["output"] = {
        "transcript": str(TXT_FILE),
        "word_count": words
    }

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_time = (
        download_time +
        load_time +
        inference_time
    )

    benchmark["total_pipeline_seconds"] = round(
        total_time, 3
    )

    with open(JSON_FILE, "w", encoding="utf-8") as f:

        json.dump(
            benchmark,
            f,
            indent=4,
            ensure_ascii=False
        )

    section("FINAL PERFORMANCE REPORT")

    log.info("Audio duration:       %.2f seconds", duration)
    log.info("Model initialization: %.2f seconds", load_time)
    log.info("Inference time:       %.2f seconds", inference_time)
    log.info("RTF:                  %.5f", rtf)
    log.info("RTFx:                 %.2fx", rtf_x)
    log.info("Words:                %s", words)

    resources = benchmark["process_resources"]

    log.info(
        "Peak process RAM:     %.2f MB",
        resources["peak_process_ram_mb"]
    )

    log.info(
        "Average CPU:          %.2f%%",
        resources["average_cpu_percent"]
    )

    log.info(
        "Peak CPU:             %.2f%%",
        resources["peak_cpu_percent"]
    )

    log.info(
        "MLX Metal memory:     %s",
        benchmark["metal_after_inference"]
    )

    log.info("Total pipeline:       %.2f seconds", total_time)

    log.info("Transcript: %s", TXT_FILE)
    log.info("JSON report: %s", JSON_FILE)
    log.info("Log file: %s", LOG_FILE)

    section("BENCHMARK COMPLETED")


if __name__ == "__main__":

    try:
        main()

    except Exception:

        log.exception("BENCHMARK FAILED")
        raise

