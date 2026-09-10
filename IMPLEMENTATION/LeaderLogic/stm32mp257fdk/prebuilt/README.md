# Prebuilt `llama-cpp-python` for the STM32MP257F-DK (Cortex-A35)

`llama_cpp_python-0.3.23-cortexa35.tar.gz` is a ready-to-use build of
`llama-cpp-python` 0.3.23 with the llama.cpp backend compiled for the
Cortex-A35. Use it to install on a new board **without pip and without
compiling anything** (pip is unreliable on these images, which is the point
of this package).

To rebuild it from source instead, see "Cross-compiling llama-cpp-python for
the STM32MP2 (Cortex-A35)" in [`IMPLEMENTATION/README.md`](../../../README.md).

## Install (on the board)

```sh
tar xzf llama_cpp_python-0.3.23-cortexa35.tar.gz -C /usr/lib/python3.12/site-packages/
```

That is the whole install. The archive contains the `llama_cpp/` package
(Python bindings + the `.so` backends in `llama_cpp/lib/`) and its
`dist-info/`, so `pip show llama-cpp-python` still reports it correctly.

Extract with `tar`, not `scp -r`: the flat library names
(`libllama.so`, `libllama.so.0`) are symlinks to the versioned files, and
copying them as real files wastes ~7 MB and breaks the versioning layout.

## Verify

```sh
python3 -c "from llama_cpp import llama_cpp; llama_cpp.llama_backend_init(); print(llama_cpp.llama_print_system_info().decode())"
```

Expected - this is the optimized build:

```
CPU : NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | REPACK = 1 |
```

`REPACK = 1` is the ARM weight-repacking path for quantized matmuls and is
where most of the speed comes from. `DOTPROD` is absent on purpose: the A35
does not implement it (`/proc/cpuinfo` shows `fp asimd evtstrm crc32 cpuid`,
no `asimddp`), so a dotprod build would crash with SIGILL.

## Requirements

| | Needed | On the DK image |
|---|---|---|
| Architecture | aarch64, NEON (no dotprod) | Cortex-A35, 2 cores |
| glibc | >= 2.39 | 2.39 |
| libstdc++ | >= 6.0.32 | 6.0.32 |
| Python | 3.12 (tested) | 3.12.12 |
| Python deps | `diskcache`, `jinja2`, `numpy`, `typing_extensions` | already present |

glibc/libstdc++/aarch64 are hard requirements - the `.so` files link against
them. Python 3.12 is what this was built and tested on; the package is pure
Python plus `ctypes` (no compiled CPython extension), so other 3.x versions
will likely work but are untested.

Only for this board family. On a Cortex-A76 (Raspberry Pi 5) this build runs
but leaves performance unused - that core has dotprod, so rebuild for it if you need to run llama.cpp on raspberry's CPU. 
However, in that case, ollama is preferred because it is much easier to install and performs comparatevely well.
If you are using the Hailo AI HAT +2, then llama.cpp is not required as you will be using the hailo platform software.

## Usage

```python
from llama_cpp import Llama
llm = Llama(model_path="model.gguf", n_threads=2)   # 2 = both A35 cores
```

Measured on the DK with `functiongemma-270m-it-Q4_K_M.gguf`, `n_threads=2`:
**~4.4-4.5 tok/s**.
