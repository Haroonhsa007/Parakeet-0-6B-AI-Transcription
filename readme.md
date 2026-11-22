# Parakeet TDT 0.6B v3
### Cross-Platform AI Speech Recognition

Run NVIDIA Parakeet TDT 0.6B v3 locally on macOS, Linux, and Windows. Supports real-time microphone transcription, live transcript updates, offline audio processing, and performance benchmarking.

<p align="center">
  <img src="https://img.shields.io/badge/Model-Parakeet%20TDT%200.6B-blue" />
  <img src="https://img.shields.io/badge/macOS-Apple%20MLX-black?logo=apple" />
  <img src="https://img.shields.io/badge/Linux-ONNX-green?logo=linux" />
  <img src="https://img.shields.io/badge/Windows-ONNX-blue?logo=windows" />
  <img src="https://img.shields.io/badge/Inference-Local-success" />
  <img src="https://img.shields.io/badge/License-See%20Model%20Terms-lightgrey" />
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#platform-support">Platforms</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#benchmark-results">Benchmarks</a> ·
  <a href="#architecture">Architecture</a>
</p>

---

## Performance at a Glance

<details>
<summary>Apple M4 Mac mini benchmark, 16 GB unified memory</summary>

| Metric | Result |
|---|---:|
| Audio duration | 371.2 seconds |
| Median transcription time | 10.25 seconds |
| Average transcription time | 10.36 seconds |
| Real-time factor | 36.23× |
| Transcribed words | 1,033 |
| Sentences | 81 |
| INT8 model file | ~720 MB |

</details>

The benchmark processed over six minutes of recorded audio in approximately ten seconds using local MLX inference.

## Features

- Local speech recognition without cloud transcription APIs.
- Real-time microphone capture.
- Live transcript updates.
- Apple Silicon acceleration through MLX.
- NVIDIA CUDA support through ONNX Runtime, subject to backend compatibility.
- CPU inference support.
- INT8 quantization for the macOS MLX checkpoint.
- Automatic model download on macOS.
- Configurable audio chunk duration.
- Transcript export to TXT.
- Repeated inference benchmarks.
- Runtime and memory monitoring.
- Separate platform-specific implementations.

## Model Information

This project uses two separate model artifacts.

| Model | Purpose |
|---|---|
| NVIDIA Parakeet TDT 0.6B v3 | Original speech recognition model |
| Sonic Speech Parakeet INT8 | Apple MLX-compatible INT8 checkpoint |

Original NVIDIA model:

https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3

Apple MLX INT8 checkpoint:

https://huggingface.co/sonic-speech/parakeet-tdt-0.6b-v3-int8

The original NVIDIA checkpoint and Apple MLX INT8 conversion are separate artifacts. They are not interchangeable.

---

## Platform Support

| Platform | Runtime | Acceleration |
|---|---|---|
| macOS Apple Silicon | Parakeet MLX | Apple Metal |
| Linux | Streaming ONNX | NVIDIA CUDA or CPU |
| Windows | Streaming ONNX | CPU by default |

Windows CUDA support depends on the installed ONNX Runtime build and upstream implementation support.

## Project Structure

```text
parakeet/
├── models/
│   └── parakeet-tdt-0.6b-v3-int8/
├── audio.wav
├── transcribe_improved.py
├── realtime_mac.py
├── realtime_linux.py
├── realtime_windows.py
├── live_transcript.txt
├── benchmark_*.json
├── benchmark_*.log
└── transcript_*.txt
```

---

# Installation

## 1. macOS, Apple Silicon

The macOS implementation uses MLX and the INT8 Parakeet checkpoint.

### Requirements

- Apple Silicon Mac.
- Python 3.
- Microphone.
- macOS microphone permissions.

### Environment setup

```bash
mkdir -p ~/AI/parakeet
cd ~/AI/parakeet

python3 -m venv .venv
source .venv/bin/activate

pip install parakeet-mlx mlx sounddevice numpy huggingface_hub
```

### Real-time transcription

Copy `realtime_mac.py` into the project directory.

```bash
python realtime_mac.py
```

The script downloads missing model files automatically.

### Configuration

```bash
python realtime_mac.py \
  --chunk-seconds 1.5 \
  --output live_transcript.txt
```

List microphone devices:

```bash
python realtime_mac.py --list-devices
```

Select a microphone:

```bash
python realtime_mac.py --device 1
```

### macOS configuration

| Parameter | Default |
|---|---|
| Model | Parakeet TDT 0.6B v3 INT8 |
| Runtime | MLX |
| Sample rate | Model-configured |
| Channels | Mono |
| Chunk duration | 1.5 seconds |
| Output | `live_transcript.txt` |

---

## 2. Linux, ONNX Runtime

The Linux implementation uses the streaming ONNX implementation.

Upstream repository:

https://github.com/dhyuk54/parakeet-streaming-onnx

### Requirements

- Ubuntu 22.04 or 24.04.
- Python version supported by upstream dependencies.
- Microphone.
- ONNX model files.
- NVIDIA GPU for CUDA acceleration, optional.

### Installation

Clone the upstream repository:

```bash
git clone https://github.com/dhyuk54/parakeet-streaming-onnx.git

cd parakeet-streaming-onnx
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install sounddevice
```

Prepare the ONNX model using the upstream repository instructions.

Copy `realtime_linux.py` into the repository root.

### CPU inference

```bash
python realtime_linux.py \
  --model-dir models/parakeet-tdt-0.6b-v3-onnx
```

### NVIDIA CUDA inference

Install a compatible GPU runtime:

```bash
pip install onnxruntime-gpu
```

Run:

```bash
python realtime_linux.py \
  --model-dir models/parakeet-tdt-0.6b-v3-onnx \
  --providers CUDAExecutionProvider CPUExecutionProvider
```

Verify execution providers:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Configuration

```bash
python realtime_linux.py \
  --chunk-seconds 2 \
  --left-context-seconds 10 \
  --right-context-seconds 2 \
  --output live_transcript.txt
```

List microphone devices:

```bash
python realtime_linux.py --list-devices
```

---

## 3. Windows, ONNX Runtime

The Windows implementation uses the same streaming ONNX implementation.

CPU execution is the default. CUDA requires a compatible Windows ONNX Runtime GPU installation and upstream backend support.

### Requirements

- Windows 10 or Windows 11.
- Python 3.11, subject to upstream dependency compatibility.
- Microphone.
- ONNX model files.

### Installation

Clone the upstream repository:

```powershell
git clone https://github.com/dhyuk54/parakeet-streaming-onnx.git

cd parakeet-streaming-onnx
```

Create a virtual environment:

```powershell
py -3.11 -m venv .venv

.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

pip install sounddevice
```

Prepare the ONNX model using upstream instructions.

Copy `realtime_windows.py` into the repository root.

### Run transcription

```powershell
python realtime_windows.py `
  --model-dir models/parakeet-tdt-0.6b-v3-onnx
```

### Configuration

```powershell
python realtime_windows.py `
  --chunk-seconds 2 `
  --left-context-seconds 10 `
  --right-context-seconds 2 `
  --output live_transcript.txt
```

List microphone devices:

```powershell
python realtime_windows.py --list-devices
```

---

# Real-Time Configuration

| Parameter | macOS | Linux | Windows |
|---|---|---|---|
| Audio input | Microphone | Microphone | Microphone |
| Sample rate | Model-configured | 16 kHz | 16 kHz |
| Channels | Mono | Mono | Mono |
| Default chunk | 1.5 sec | 2 sec | 2 sec |
| Left context | MLX-managed | 10 sec | 10 sec |
| Right context | MLX-managed | 2 sec | 2 sec |
| Transcript output | TXT | TXT | TXT |

Linux and Windows use chunk-based processing. Approximate algorithmic latency depends on chunk duration and right-context duration.

End-to-end latency also depends on inference speed, microphone buffering, hardware, and system load.

---

# Architecture

```text
                  MICROPHONE
                      |
                      v
                AUDIO CAPTURE
                      |
                      v
                AUDIO BUFFER
                      |
              16 kHz MONO PCM
                      |
                      v
                STREAMING ASR
                      |
                      v
              TRANSCRIPT PROCESSING
                      |
              +-------+-------+
              |               |
              v               v
         LIVE TEXT         TXT EXPORT
```

### Platform-specific execution

```text
                  AUDIO INPUT
                       |
                       v
               PLATFORM DETECTION
                       |
           +-----------+-----------+
           |           |           |
           v           v           v
         macOS       Linux      Windows
           |           |           |
           v           v           v
          MLX       ONNX Runtime  ONNX Runtime
           |           |           |
           v           v           v
         Metal    CUDA / CPU      CPU
           |           |           |
           +-----------+-----------+
                       |
                       v
                TRANSCRIPT OUTPUT
```

## Suggested Production Architecture

| Component | Suggested Technology |
|---|---|
| Audio transport | WebRTC / WebSocket |
| Audio capture | sounddevice |
| Voice activity detection | Silero VAD |
| Speech recognition | Parakeet |
| Backend | FastAPI |
| Live updates | WebSocket |
| Background execution | Async workers |
| Deployment | Docker |

---

# Benchmark Results

## Hardware

| Component | Specification |
|---|---|
| Device | Apple Mac mini |
| Processor | Apple M4 |
| Unified memory | 16 GB |
| Runtime | MLX |
| Quantization | INT8 |
| Audio format | Mono WAV, PCM 16-bit |
| Audio sample rate | 44.1 kHz |

## Full Audio Benchmark

Audio duration: 371.2 seconds.

| Metric | Result |
|---|---:|
| Median inference time | 10.25 sec |
| Average inference time | 10.36 sec |
| P95 inference time | 10.77 sec |
| Real-time factor | 36.23× |
| Words | 1,033 |
| Sentences | 81 |
| Model file size | ~720 MB |

## Short Audio Tests

| Audio duration | Processing time | RTFx |
|---|---:|---:|
| 30 seconds | 0.64 sec | 46.62× |
| 60 seconds | 1.27 sec | 47.06× |
| 120 seconds | 2.68 sec | 44.83× |

These results represent offline file transcription. They do not represent real-time microphone latency.

No equivalent Linux or Windows benchmark results have been collected yet.

---

# Benchmark Methodology

The offline benchmark measures:

- Model initialization time.
- Warm-up duration.
- Repeated inference time.
- Median and average latency.
- P95 latency.
- Standard deviation.
- Real-time factor.
- CPU and process memory.
- System RAM availability.
- Swap usage.
- MLX Metal memory statistics.

### Real-Time Factor

```text
RTFx = Audio Duration / Processing Time
```

Higher RTFx means faster transcription relative to audio duration.

### Accuracy Evaluation

Transcription speed alone does not establish recognition accuracy.

Recommended evaluation metrics:

- Word Error Rate (WER).
- Character Error Rate (CER).
- Word-level timestamp accuracy.
- Domain terminology accuracy.
- Partial transcript stability.
- Live transcription latency.

Use a human-verified reference transcript for accuracy measurements.

---

# Implementation Notes

- macOS uses an MLX-compatible INT8 checkpoint.
- Linux and Windows require an ONNX model prepared using the upstream streaming repository.
- Linux and Windows depend on the upstream `StreamingTdtASR` Python interface.
- Windows defaults to CPU execution.
- CUDA requires a compatible runtime and backend.
- Chunk-based transcription produces periodic updates.
- These scripts do not provide native token-by-token streaming.
- Syntax validation does not establish successful runtime operation on every target platform.

---

# References

| Project | Resource |
|---|---|
| NVIDIA Parakeet | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| Apple MLX | https://github.com/ml-explore/mlx |
| Parakeet MLX INT8 | https://huggingface.co/sonic-speech/parakeet-tdt-0.6b-v3-int8 |
| Streaming ONNX | https://github.com/dhyuk54/parakeet-streaming-onnx |
| NVIDIA NeMo | https://github.com/NVIDIA/NeMo |
| ONNX Runtime | https://onnxruntime.ai/ |

## License

The original NVIDIA Parakeet TDT v3 model uses CC BY 4.0.

Review the licenses of converted checkpoints, dependencies, and runtime implementations before redistribution or commercial deployment.

---

<p align="center">
  <strong>Local AI. Cross-platform speech recognition.</strong>
  <br />
  Built and benchmarked by Haroon Saleem.
  <br />
  <a href="https://github.com/Haroonhsa007">GitHub</a>
</p>