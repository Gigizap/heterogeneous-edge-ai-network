# RPi 5 + Hailo-10H Benchmark Suite

Benchmarks for Qwen3-1.7B (LLM) and YOLOv8n (object detection) on Raspberry Pi 5 CPU and Hailo-10H accelerator.

## Hardware

- Raspberry Pi 5
- Hailo AI HAT+ 2 (Hailo H10 chip), HailoRT ≥ 5.30
- Raspberry Pi Camera Module 2
- FNIRSI FNB-C2 USB power meter (for power consumption measurement)

## Setup

```bash
pip install -r requirements.txt
```

`hailo-platform` is **not** available on PyPI. Install the Hailo runtime, driver, and GenAI packages by following the official [Raspberry Pi AI software documentation](https://www.raspberrypi.com/documentation/computers/ai.html) (see the AI HAT+ 2 section).

## Models

Download these and place them in the `test_raspberrypihat` folder.

### Qwen3-1.7B

| File | Target | Source |
|---|---|---|
| `Qwen3-1.7B-Instruct.hef` | Hailo | [Hailo Model Explorer](https://hailo.ai/products/hailo-software/model-explorer/generative-ai/qwen3-1-7b-instruct/) |
| `Qwen3-1.7B-Q4_K_M.gguf` | CPU | [unsloth/Qwen3-1.7B-GGUF](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF) (Q4_K_M variant) |

Note: the Hailo website references the base model ([Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)), not the Instruct variant.

### YOLOv8n

| File | Target | Source |
|---|---|---|
| `yolov8n.hef` | Hailo | [Hailo Model Zoo - HAILO10H object detection](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst) |
| `yolov8n.onnx` | CPU | Same link above |

## Usage

1. Attach the FNIRSI FNB-C2 to the Pi's USB-C power input to measure power consumption.
2. Run the launcher:

```bash
python3 LAUNCHER.py
```

3. Pick a combo from the menu. Results are saved to `avg_*.txt` files after 20 minutes or on Ctrl+C.

## Scripts

| Script | What it does |
|---|---|
| `qwen3_cpu.py` | Qwen3 inference on CPU (llama.cpp) |
| `qwen3_hailo.py` | Qwen3 inference on Hailo-10H |
| `detect_cpu.py` | YOLOv8n detection on CPU (ONNX Runtime) |
| `detect_hailo.py` | YOLOv8n detection on Hailo-10H |
| `run_leader.py` | P2P discovery/messaging leader node |
