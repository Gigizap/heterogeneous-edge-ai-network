# Prebuilt `llama-cpp-python` for the STM32MP257F-DK (Cortex-A35)

`llama_cpp_python-0.3.23-cortexa35.tar.gz` is a ready-to-use build of
`llama-cpp-python` 0.3.23 with the llama.cpp backend compiled for the
Cortex-A35. Use it to install on a new board **without pip and without
compiling anything** (pip is unreliable on these images, which is the point
of this package).

To install it on a board, follow step 5 of the board setup guide, [`../README.md`](../README.md): it covers the extract, the two Python dependencies and the linker configuration. Building a new archive is described below.

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
| Python deps | `diskcache`, `jinja2`, `numpy`, `typing_extensions` | `apt-get install python3-diskcache python3-jinja2` |

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

## Building it yourself

Only needed to rebuild the archive: a new llama.cpp version, a different Python, or another CPU. A plain `pip install` on the board builds without NEON or runs out of memory, so the wheel is cross-compiled on an x86_64 host.

Target: OpenSTLinux `5.0.15-...-scarthgap-mpu-v26.02.18`, Cortex-A35 (AArch64).

### 1. Get the SDK

Download (needs free myST login + accept SLA0048):
- Page: https://wiki.st.com/stm32mpu/wiki/STM32MPU_Developer_Package
- File: `SDK-x86_64-stm32mp2-openstlinux-6.6-yocto-scarthgap-mpu-v26.02.18.tar.gz`

Install guide: https://wiki.st.com/stm32mpu/wiki/Getting_started/STM32MP2_boards/STM32MP257x-DK/Develop_on_Arm_Cortex-A35/Install_the_SDK

### 2. Install the SDK (host)

```bash
tar xf SDK-x86_64-stm32mp2-openstlinux-6.6-yocto-scarthgap-mpu-v26.02.18.tar.gz
cd SDK-x86_64-stm32mp2-*
./st-image-weston-*-toolchain-5.0.15-*.sh     # accept default install dir /opt/st/...
```

### 3. Activate the toolchain (host)

```bash
source /opt/st/.../environment-setup-cortexa35-ostl-linux
```

Verify (both must succeed):

```bash
echo $OECORE_SDK_VERSION   # -> 5.0.15-openstlinux-6.6-yocto-scarthgap-mpu-v26.02.18
$CC --version              # -> ST cross-gcc
```

### 4. Check Python versions match

```bash
# on the BOARD:
python3 --version
```

The host Python that runs `pip wheel` must be the SAME minor version (e.g. 3.12). Wheels are not interchangeable across minor versions.

### 5. Build the wheel (host)

```bash
CMAKE_ARGS="-DGGML_NATIVE=OFF \
  -DGGML_NEON=ON \
  -DGGML_ARM_FMA=ON \
  -DGGML_F16C=OFF \
  -DGGML_AVX=OFF \
  -DGGML_AVX2=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_FMA=OFF \
  -DGGML_SSE3=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_C_FLAGS='-mcpu=cortex-a35+crc -O3 -ffast-math -fno-finite-math-only' \
  -DCMAKE_CXX_FLAGS='-mcpu=cortex-a35+crc -O3 -ffast-math -fno-finite-math-only'" \
FORCE_CMAKE=1 \
pip wheel llama-cpp-python --no-deps -w ./dist
```

Output: `./dist/llama_cpp_python-*.whl`

### 6. Copy to the board

```bash
scp ./dist/llama_cpp_python-*.whl root@<board-ip>:/tmp/
```

Then install it as in step 5 of [`../README.md`](../README.md).

### Notes

- `FORCE_CMAKE=1` makes the package build its OWN bundled llama.cpp with these flags. No separate backend needed.
- `n_threads=2` = the `-t 2` that doubles throughput. Don't go higher (only 2 cores).
- `-DLLAMA_CURL=ON` needs libcurl in the sysroot (present in the ST SDK). If configure fails on curl, that's why.
- Building on the host avoids the on-board OOM (the device can't compile this in its RAM).
