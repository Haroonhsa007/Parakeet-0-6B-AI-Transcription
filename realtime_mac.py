#!/usr/bin/env python3
"""
Chunked live microphone transcription for Apple Silicon using Parakeet MLX.

This deliberately uses model.transcribe() on short overlapping WAV chunks.
It does NOT call model.transcribe_stream(), which is incompatible with the
INT8 checkpoint/runtime combination reported by some parakeet-mlx versions.

Install:
  pip install -U parakeet-mlx mlx sounddevice numpy huggingface_hub

Run:
  python realtime_mac.py --device 1
  python realtime_mac.py --device 1 --chunk-seconds 3 --overlap-seconds 0.6

This is chunked near-real-time transcription, not native token streaming.
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import tempfile
import wave
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sounddevice as sd
from huggingface_hub import hf_hub_download
from parakeet_mlx.utils import from_config


def parse_args():
    p = argparse.ArgumentParser(description="Chunked live Parakeet transcription for macOS")
    p.add_argument("--model", default="sonic-speech/parakeet-tdt-0.6b-v3-int8")
    p.add_argument("--chunk-seconds", type=float, default=3.0)
    p.add_argument("--overlap-seconds", type=float, default=0.6)
    p.add_argument("--device", type=int, default=None, help="Input device index")
    p.add_argument("--output", default="live_transcript.txt")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def load_model(repo_id: str):
    local_dir = Path("models") / repo_id.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for filename in ("config.json", "model.safetensors", "quantization_config.json"):
        paths[filename] = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )

    with open(paths["config.json"], encoding="utf-8") as f:
        config = json.load(f)
    with open(paths["quantization_config.json"], encoding="utf-8") as f:
        qconfig = json.load(f)

    model = from_config(config)
    nn.quantize(
        model,
        bits=int(qconfig["bits"]),
        group_size=int(qconfig["group_size"]),
        class_predicate=lambda path, layer: (
            path.startswith("encoder.") and isinstance(layer, nn.Linear)
        ),
    )
    model.load_weights(paths["model.safetensors"])
    mx.eval(model.parameters())
    mx.synchronize()
    return model


def write_wav(path: Path, audio: np.ndarray, sample_rate: int):
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def get_text(model, audio: np.ndarray, sample_rate: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        write_wav(wav_path, audio, sample_rate)
        result = model.transcribe(str(wav_path))
        return (result.text or "").strip()
    finally:
        wav_path.unlink(missing_ok=True)


def merge_chunk_text(existing: str, new_text: str, max_overlap_words: int = 12) -> str:
    """Remove repeated words caused by adjacent audio-window overlap."""
    old_words = existing.split()
    new_words = new_text.split()
    if not old_words:
        return new_text
    if not new_words:
        return existing

    old_norm = [w.strip(".,!?;:").lower() for w in old_words]
    new_norm = [w.strip(".,!?;:").lower() for w in new_words]

    best = 0
    limit = min(max_overlap_words, len(old_words), len(new_words))
    for n in range(1, limit + 1):
        if old_norm[-n:] == new_norm[:n]:
            best = n

    return " ".join(old_words + new_words[best:]).strip()


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be greater than zero")
    if args.overlap_seconds < 0 or args.overlap_seconds >= args.chunk_seconds:
        raise SystemExit("--overlap-seconds must be >= 0 and less than --chunk-seconds")

    print(f"Loading INT8 checkpoint: {args.model}")
    model = load_model(args.model)
    sample_rate = int(model.preprocessor_config.sample_rate)

    chunk_samples = int(sample_rate * args.chunk_seconds)
    overlap_samples = int(sample_rate * args.overlap_seconds)
    step_samples = chunk_samples - overlap_samples

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
    stop_requested = False
    transcript = ""
    pending = np.empty(0, dtype=np.float32)

    def audio_callback(indata, frames, callback_time, status):
        if status:
            print(f"\nAudio input warning: {status}", file=sys.stderr)
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            print("\nAudio queue full. Audio block dropped.", file=sys.stderr)

    def stop_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)

    def transcribe_chunk(chunk: np.ndarray):
        nonlocal transcript
        if len(chunk) < int(sample_rate * 0.5):
            return
        text = get_text(model, chunk, sample_rate)
        if text:
            transcript = merge_chunk_text(transcript, text)
            print("\n" + transcript, flush=True)
            Path(args.output).write_text(transcript + "\n", encoding="utf-8")

    print(f"Microphone ready. Sample rate: {sample_rate} Hz")
    print(f"Chunk: {args.chunk_seconds:.1f}s, overlap: {args.overlap_seconds:.1f}s")
    print("Speak. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=args.device,
            callback=audio_callback,
            blocksize=0,
        ):
            while not stop_requested:
                try:
                    block = audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                pending = np.concatenate((pending, block))
                while len(pending) >= chunk_samples:
                    chunk = pending[:chunk_samples]
                    transcribe_chunk(chunk)
                    pending = pending[step_samples:]

    except KeyboardInterrupt:
        stop_requested = True
    finally:
        # Drain audio already queued by PortAudio before processing the tail.
        while True:
            try:
                pending = np.concatenate((pending, audio_queue.get_nowait()))
            except queue.Empty:
                break

        if len(pending) > int(sample_rate * 0.5):
            transcribe_chunk(pending)

        if transcript:
            print(f"\nSaved transcript: {Path(args.output).resolve()}")
        else:
            print("\nNo transcript text was produced.")


if __name__ == "__main__":
    main()
