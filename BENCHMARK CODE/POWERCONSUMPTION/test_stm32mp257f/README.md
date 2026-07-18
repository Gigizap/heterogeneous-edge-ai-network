# STM32MP257F-DK Benchmark Suite

Benchmarks for FunctionGemma-270M (LLM) and YOLOv8n person detection on the STM32MP257F-DK CPU and on-chip NPU.

## Hardware

- STM32MP257F-DK
- On-chip NPU (via ST Edge AI / `stai_mpu`)
- Camera for person detection
- FNIRSI FNB-C2 USB power meter (for power consumption measurement)

## Setup

```bash
pip install -r requirements.txt
```

The NPU runtime (`stai_mpu`, TFLite) is installed via the OpenSTLinux `x-linux-ai` packages, not from PyPI.

## Models

The YOLOv8n person-detection models ship in this folder:

| File | Target |
|---|---|
| `yolov8n_320_quant_pt_uf_od_coco-person-st.tflite` | CPU |
| `yolov8n_320_quant_pt_uf_od_coco-person-st.nb` | NPU |

FunctionGemma-270M for the CPU (llama.cpp) is not shipped here: place `functiongemma-270m-it-Q4_K_M.gguf` in this folder before running.

## Usage

1. Attach the FNIRSI FNB-C2 to the board's USB-C power input.
2. Run the launcher:

```bash
python3 LAUNCHER.py
```

3. It runs each config for 21 minutes and auto-advances. Results are saved to `avg_*.txt` files.

## Scripts

| Script | What it does |
|---|---|
| `functiongemma_cpu.py` | FunctionGemma inference on CPU (llama.cpp) |
| `detect_person_cpu.py` | YOLOv8n person detection on CPU (TFLite) |
| `detect_person_npu.py` | YOLOv8n person detection on the NPU |
| `run_leader.py` | P2P discovery/messaging leader node |
