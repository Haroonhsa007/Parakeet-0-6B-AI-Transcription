#!/usr/bin/env python3
"""
Live microphone transcription on Windows using the cross-platform
parakeet-streaming-onnx project.

Run this script from the root of a clone of:
https://github.com/dhyuk54/parakeet-streaming-onnx

The upstream project's documented CUDA provider is Linux-only. This Windows
script defaults to CPUExecutionProvider. Do not select CUDA unless your
specific ONNX Runtime build and the upstream project support your setup.

Install dependencies in PowerShell:
  py -3.11 -m venv .venv
  .venv\\Scripts\\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install sounddevice

Run:
  python realtime_windows.py --model-dir models/parakeet-tdt-0.6b-v3-onnx
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from streaming.streaming_asr import StreamingTdtASR


def parse_args():
    p = argparse.ArgumentParser(description="Live Parakeet transcription for Windows")
    p.add_argument("--model-dir", default="models/parakeet-tdt-0.6b-v3-onnx")
    p.add_argument("--providers", nargs="+",
                   default=["CPUExecutionProvider"],
                   help="ONNX Runtime providers")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--chunk-seconds", type=float, default=2.0)
    p.add_argument("--left-context-seconds", type=float, default=10.0)
    p.add_argument("--right-context-seconds", type=float, default=2.0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--output", default="live_transcript.txt")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be greater than zero")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    import onnxruntime as ort
    available = ort.get_available_providers()
    providers = [x for x in args.providers if x in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    print("Available ONNX providers:", available)
    print("Using ONNX providers:", providers)

    asr = StreamingTdtASR(
        model_dir=str(model_dir),
        chunk_secs=args.chunk_seconds,
        left_context_secs=args.left_context_seconds,
        right_context_secs=args.right_context_seconds,
        providers=providers,
    )

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
    stop = False
    chunk_samples = int(args.sample_rate * args.chunk_seconds)
    pending = np.empty(0, dtype=np.float32)
    last_text = ""

    def audio_callback(indata, frames, callback_time, status):
        if status:
            print(f"\nAudio input: {status}", file=sys.stderr)
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            print("\nAudio queue full, dropping an input block.", file=sys.stderr)

    def stop_handler(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)

    print(f"Microphone ready at {args.sample_rate} Hz.")
    print(f"Approximate algorithmic latency: "
          f"{args.chunk_seconds + args.right_context_seconds:.1f}s")
    print("Speak. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            device=args.device,
            callback=audio_callback,
            blocksize=0,
        ):
            while not stop:
                try:
                    block = audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                pending = np.concatenate((pending, block))
                while len(pending) >= chunk_samples:
                    audio_chunk = pending[:chunk_samples]
                    pending = pending[chunk_samples:]
                    asr.process_chunk(audio_chunk)
                    text = asr.get_full_text().strip()
                    if text and text != last_text:
                        print("\r" + text + " " * 8, end="", flush=True)
                        last_text = text

    except KeyboardInterrupt:
        pass
    finally:
        final_text = asr.get_full_text().strip()
        if final_text:
            Path(args.output).write_text(final_text + "\n", encoding="utf-8")
            print("\n\nSaved transcript:", Path(args.output).resolve())
        else:
            print("\nNo transcript text was produced.")


if __name__ == "__main__":
    main()
