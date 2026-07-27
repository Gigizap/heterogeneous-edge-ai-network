# Benchmarks

## Accuracy tests (Part A, B, C)

Measuring accuracy of tool call and grade replies.
See [ACCURACY/README.md](ACCURACY/README.md).

## FunctionGemma test handlers

Separate comparison of grammar-enforcing handlers for accuracy and latency.
See [FUNCTIONGEMMA TEST HANDLERS/README.md](FUNCTIONGEMMA%20TEST%20HANDLERS/README.md).

## Power consumption and inference performance

Measures power draw and inference throughput (FPS, tokens/s) on two edge boards: Raspberry Pi 5 with Hailo AI HAT+ 2 and STM32MP257F-DK. Running LLM and object detection workloads solo and in parallel to test different configurations.
See [POWERCONSUMPTION/README.md](POWERCONSUMPTION/README.md).

## Latency and network

Leader answer latency (reply-count and tool-count sweeps, TTFT) and network-overlay scaling (discovery + tool aggregation as the fleet grows), plus the analysis scripts and result sets behind the latency and network report sections.
See [LATENCY AND NETWORK/README.md](LATENCY%20AND%20NETWORK/README.md).

## Image detection sanity check

Runs both edge YOLO models (ONNX 640px / 80-class COCO, and the TFLite 320px person-only model) on a laptop against local images, using the same preprocessing the boards use.
See [test/README.md](test/README.md); run [`test_images.py`](test_images.py).

