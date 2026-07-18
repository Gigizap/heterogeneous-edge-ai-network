# test images folder

Drop any images here (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`), then from
the repo root run:

```bash
python test_images.py
```

It runs both edge YOLO models (ONNX 640px / 80-class COCO, and the TFLite 320px
person-only model) using the exact same preprocessing the boards use, and writes
annotated copies to `test/results/`. See the header of `test_images.py` for
options and the pip packages you need.
