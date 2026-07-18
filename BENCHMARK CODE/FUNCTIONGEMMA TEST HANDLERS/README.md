# functiongemma test handlers

This compares accuracy and latency of the grammar-enforcing handlers (used in the related works section).

## setup

Install requirements:
```
pip install -r requirements.txt
```

> **Note:** `llama-cpp-python` should to be built with CUDA support for GPU inference.

Put the functiongemma model in the `models/` folder.  
Model name must be: `functiongemma-270m-it-Q4_K_M.gguf`

## running

1. Run `test_handlers_functiongemma.py`
2. Run `plot.py`
3. Check results in `fg_compare_out/`