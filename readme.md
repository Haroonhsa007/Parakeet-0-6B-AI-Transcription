# Parakeet TDT 0.6B v3, Cross-Platform Speech Recognition

Local speech-to-text benchmarking and real-time microphone transcription using NVIDIA Parakeet TDT 0.6B v3.

Supports separate deployment paths for macOS, Linux, and Windows.

## Features

- Local speech recognition.
- Real-time microphone capture.
- Live transcript updates.
- Apple Silicon MLX acceleration.
- NVIDIA CUDA support through ONNX Runtime.
- CPU inference support.
- INT8 quantization on the macOS MLX checkpoint.
- Automatic model download on macOS.
- Configurable audio chunk duration.
- Transcript export to TXT.
- Benchmark measurements and memory monitoring.

## Model

Original NVIDIA model:

https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3

Apple MLX INT8 checkpoint:

https://huggingface.co/sonic-speech/parakeet-tdt-0.6b-v3-int8

The original NVIDIA checkpoint and Apple MLX INT8 conversion are separate model artifacts.

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

## Platform Support

| Platform | Runtime | Acceleration |
|---|---|---|
| macOS Apple Silicon | Parakeet MLX | Apple Metal |
| Linux | Streaming ONNX | NVIDIA CUDA or CPU |
| Windows | Streaming ONNX | CPU by default |

Windows CUDA support depends on the installed ONNX Runtime build and the upstream implementation.

---

# 1. macOS Real-Time Transcription

Uses your Apple Silicon Mac and the INT8 MLX checkpoint.

### Requirements

- Apple Silicon Mac.
- Python 3.
- Microphone.
- macOS microphone permission.

### Installation

```bash
mkdir -p ~/AI/parakeet
cd ~/AI/parakeet

python3 -m venv .venv
source .venv/bin/activate

pip install parakeet-mlx mlx sounddevice numpy huggingface_hub
```

### Run

Download or copy `realtime_mac.py` into the project directory.

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

### macOS Settings

| Parameter | Default |
|---|---|
| Model | Parakeet TDT 0.6B v3 INT8 |
| Runtime | MLX |
| Sample rate | Model-configured |
| Channels | Mono |
| Chunk duration | 1.5 seconds |
| Output | live_transcript.txt |

---

# 2. Linux Real-Time Transcription

Uses the streaming ONNX implementation.

Repository:

https://github.com/dhyuk54/parakeet-streaming-onnx

### Requirements

- Ubuntu 22.04 or 24.04.
- Python version supported by the upstream project.
- Microphone.
- ONNX model files.
- NVIDIA GPU for CUDA acceleration, optional.

### Installation

Clone the upstream repository:

```bash
git clone https://github.com/dhyuk54/parakeet-streaming-onnx.git

cd parakeet-streaming-onnx
```

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install sounddevice
```

Prepare the ONNX model following the upstream repository instructions.

Copy `realtime_linux.py` into the repository root.

### Run with CPU

```bash
python realtime_linux.py \
  --model-dir models/parakeet-tdt-0.6b-v3-onnx
```

### Run with NVIDIA CUDA

Install a compatible ONNX Runtime GPU package:

```bash
pip install onnxruntime-gpu
```

Run:

```bash
python realtime_linux.py \
  --model-dir models/parakeet-tdt-0.6b-v3-onnx \
  --providers CUDAExecutionProvider CPUExecutionProvider
```

Verify available execution providers:

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

List microphones:

```bash
python realtime_linux.py --list-devices
```

---

# 3. Windows Real-Time Transcription

Uses the same streaming ONNX implementation.

Windows execution defaults to CPU. CUDA support requires a compatible Windows ONNX Runtime GPU build and upstream backend support.

### Requirements

- Windows 10 or Windows 11.
- Python 3.11, subject to upstream dependency compatibility.
- Microphone.
- ONNX model files.

### Installation

Clone the repository:

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

Prepare the ONNX model using the upstream repository instructions.

Copy `realtime_windows.py` into the repository root.

### Run

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

# 4. Real-Time Configuration

| Parameter | macOS | Linux | Windows |
|---|---:|---:|---:|
| Audio input | Microphone | Microphone | Microphone |
| Sample rate | Model-configured | 16 kHz | 16 kHz |
| Channels | Mono | Mono | Mono |
| Default chunk | 1.5 sec | 2 sec | 2 sec |
| Left context | MLX-managed | 10 sec | 10 sec |
| Right context | MLX-managed | 2 sec | 2 sec |
| Transcript output | TXT | TXT | TXT |

The Linux and Windows scripts use chunk-based processing. Their approximate algorithmic latency is influenced by chunk duration and right-context duration.

Actual end-to-end latency also depends on model inference speed, microphone buffering, hardware, and system load.

---

# 5. Real-Time Architecture

```text
Microphone
    |
Audio Capture
    |
16 kHz Mono PCM
    |
Audio Buffer
    |
Streaming ASR
    |
Transcript Processing
    |
Live Text Output
    |
TXT Export
```

For production applications:

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

# 6. Offline Benchmark Results

Hardware: Apple M4 Mac mini, 16 GB unified memory.

Audio duration: 371.2 seconds.

| Metric | Result |
|---|---:|
| Median inference time | 10.25 sec |
| Average inference time | 10.36 sec |
| Real-time factor | 36.23× |
| Words | 1,033 |
| Sentences | 81 |
| Model size | ~720 MB |

### Short Audio Tests

| Audio duration | Processing time | RTFx |
|---|---:|---:|
| 30 seconds | 0.64 sec | 46.62× |
| 60 seconds | 1.27 sec | 47.06× |
| 120 seconds | 2.68 sec | 44.83× |

These are offline file-transcription results, not live microphone latency measurements.

No equivalent Linux or Windows benchmark results have been collected yet.

---

# 7. Benchmark Methodology

The offline benchmark records:

- Model initialization time.
- Warm-up duration.
- Repeated inference time.
- Median and average latency.
- P95 latency.
- Standard deviation.
- Real-time factor.
- CPU and process memory.
- System RAM and swap.
- MLX Metal memory statistics.

Formula:

```text
RTFx = Audio Duration / Processing Time
```

Higher RTFx means faster transcription.

---

# 8. Accuracy Evaluation

Speed does not establish transcription accuracy.

Recommended evaluation:

- Word Error Rate (WER).
- Character Error Rate (CER).
- Word-level timestamp accuracy.
- Domain terminology accuracy.
- Partial transcript stability.
- Live transcription latency.

Use a human-verified reference transcript for accuracy measurements.

---

# 9. Important Implementation Notes

- The macOS script uses an MLX-compatible INT8 checkpoint.
- The Linux and Windows scripts require an ONNX model prepared using the upstream streaming repository.
- The Linux and Windows scripts depend on the upstream `StreamingTdtASR` Python interface.
- Windows defaults to CPU execution.
- CUDA execution requires a compatible runtime and backend.
- Chunk-based transcription produces periodic updates. Do not treat these scripts as native token-by-token streaming.
- The scripts have passed syntax validation, but require runtime testing on their target operating systems.

## References

- NVIDIA Parakeet: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Apple MLX: https://github.com/ml-explore/mlx
- Parakeet MLX: https://huggingface.co/sonic-speech/parakeet-tdt-0.6b-v3-int8
- Streaming ONNX: https://github.com/dhyuk54/parakeet-streaming-onnx
- NVIDIA NeMo: https://github.com/NVIDIA/NeMo
- ONNX Runtime: https://onnxruntime.ai/

## License

The original NVIDIA Parakeet TDT v3 model uses CC BY 4.0. Review the licenses of converted checkpoints, dependencies, and runtime implementations before redistribution or commercial deployment.