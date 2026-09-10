# STM32MP257F-DK setup, from the box to a running agent

Full bring-up of a board: assemble it, flash the OS, install the AI runtime, copy the code, install llama.cpp, run it. Follow the six steps in order.

Everything here was done on **ecosystem v6.2.0** (image `v26.02.18`, kernel 6.6.116). Use that version: the prebuilt llama.cpp package in [`prebuilt/`](prebuilt/) is built for it. A newer release (v6.2.1) exists but is untested with this project.

**How dependencies get installed here:** whatever is in the OpenSTLinux repos comes from `apt-get` (and the AI runtime from `x-linux-ai`); anything else comes from pip. `llama-cpp-python` is the exception: it has to be compiled for the Cortex-A35, so it ships prebuilt in [`prebuilt/`](prebuilt/) and is installed by extracting it.

---

## 1. Assembly

Follow [Assembling the board](https://wiki.stmicroelectronics.cn/stm32mpu/wiki/STM32MP25_Discovery_kits_-_Starter_Package#Assembling_the_board) on the ST wiki.

You need the board, its microSD card, a USB-C cable, and a USB-C power supply rated **5V-3A**. A weaker supply makes the LEDs next to the power port turn red or orange, and the board will not run properly.

**Connect the camera** (MB1854 board, from the B-CAMS-IMX package) as shown in that article: it plugs into CN8 on the board via the flat cable. The sensing presets need it.

The LVDS display is optional. The agent runs headless.

## 2. Flash the image

### Download

1. Go to https://www.st.com/en/embedded-software/stm32mp2starter.html
2. Click **MP2-START-A35-TD**
3. Choose version **6.2.0**
4. Click **Download as a guest**, enter your email, and open the link they send you
5. Download `FLASH-stm32mp2-openstlinux-6.6-yocto-scarthgap-mpu-v26.02.18.tar.gz` (~666 MB)

Unpack it anywhere (7-Zip and WinRAR both work on Windows):

```sh
tar xvf FLASH-stm32mp2-openstlinux-6.6-yocto-scarthgap-mpu-v26.02.18.tar.gz
```

### Install the flashing tool

Download **STM32CubeProgrammer** (v2.23.0 used here) from https://www.st.com/en/development-tools/stm32cubeprog.html and run the installer. On Windows it installs to:

```
C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\
```

The CLI is `STM32_Programmer_CLI.exe` in that folder's `bin`. Check it works:

```sh
cd "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin"
.\STM32_Programmer_CLI.exe --h
```

It prints `STM32CubeProgrammer v2.23.0` and its help.

The commands below call the `.exe` by full path, so nothing else is required. Alternatively add that `bin` folder to `PATH` and call `STM32_Programmer_CLI` from anywhere.

### Connect the board for flashing

1. Set **all four boot switches to OFF** (down). The switches are two blocks next to the microSD slot: SW1 holds BOOT0 and BOOT1, SW2 holds BOOT2 and BOOT3, with `ON` printed at the top of each block, so OFF is down. A new board ships with BOOT0 ON, so flip that one down
2. Put the microSD card in its slot
3. Plug a USB-C cable from the **OTG** port to your PC
4. Plug the **5V-3A power supply** into the power port
5. Press the **reset** button

Check the PC sees it:

```sh
.\STM32_Programmer_CLI.exe -l usb
```

Expect `Total number of available STM32 device in DFU mode: 1`. If it says `0`, recheck the switches. On Windows you may also need to run `STM32 Bootloader.bat` from the CubeProgrammer `Drivers\DFU_Driver` folder.

### Flash

Go to the unpacked image folder and write the flash layout:

```sh
cd /d "<where you unpacked>\stm32mp2-openstlinux-6.6-yocto-scarthgap-mpu-v26.02.18\images\stm32mp2"
"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe" -c port=usb1 -w flashlayout_st-image-weston\extensible\FlashLayout_sdcard_stm32mp257f-dk-extensible.tsv
```

Takes about 10 minutes; the root filesystem alone is 752 MB and runs at roughly 1.7 MB/s over USB. Do not unplug or reset during it. It ends with `Flashing service completed successfully`.

> **Use the `extensible` layout**, as above. The default layout caps the root filesystem at about 4 GB whatever the card size and parks the rest in a separate partition mounted at `/usr/local`. Models and Python packages install into the root filesystem, so with the default it fills up while the big partition stays empty.

### Boot

Set **BOOT0 back to ON** (up), leaving BOOT1, BOOT2 and BOOT3 OFF. This is boot from microSD. Press **reset**, and the board comes up after a few seconds.

Sources: [Starter Package images](https://wiki.st.com/stm32mpu/wiki/STM32MP2_Starter_Package_-_images#Archives), [Discovery kit Starter Package](https://wiki.stmicroelectronics.cn/stm32mpu/wiki/STM32MP25_Discovery_kits_-_Starter_Package).

## 3. AI runtime (X-LINUX-AI)

Only needed if the board runs a **sensing** preset (camera, object/person detection). Skip it for a leader-only board.

The board needs an internet connection first: Ethernet, or WiFi by following section 3, *Automatic WiFi configuration at start up*, of [How to setup a WLAN connection](https://wiki.st.com/stm32mpu/wiki/How_to_setup_a_WLAN_connection#Automatic_WiFi_configuration_at_start_up). Be careful to use the correct lowercase and uppercase letters!

```sh
apt-get update
apt-get install x-linux-ai-tool
x-linux-ai -v                            # check the version, e.g. 6.2.0

apt-get install python3-numpy python3-opencv
x-linux-ai -i python3-libstai-mpu        # stai_mpu module, NPU presets
x-linux-ai -i stai-mpu-tflite            # TFLite plugin, CPU presets

apt-get install stai-mpu-ovx             # OpenVX backend, required to run .nb models on the NPU
```

Without `stai-mpu-ovx` the NPU skills fail at call time with `Could not load nbg model` and `dlopen failed: /usr/lib/libstai_mpu_ovx.so.6`.

These are OpenSTLinux system packages, not pip packages. Each preset's `requirements.txt` under [`SensingLogic/`](../../SensingLogic/) lists what it needs.

References: [X-LINUX-AI expansion package](https://wiki.st.com/stm32mpu/wiki/Category:X-LINUX-AI_expansion_package#X-LINUX-AI_package), [STAI MPU Python API](https://wiki.st.com/stm32mpu/wiki/How_to_run_inference_using_the_STAI_MPU_Python_API).

## 4. Copy the code to the board

`git` is not installed on the board, so clone on your **host PC** first and copy only `IMPLEMENTATION/` across. That also avoids sending the benchmark data and figures, which the board does not need.

On the host PC:

```sh
git clone <repository URL>
```

First get the board's IP address. On the board:

```sh
ifconfig
```

Look for the `wlan0` entry (or `eth0` on Ethernet) and take the address after `inet`.

Then copy the folder over from the host PC, using that address:

```sh
scp -r <path to repo>/IMPLEMENTATION root@<board-ip>:/home/weston/
```

This gives you `/home/weston/IMPLEMENTATION/` on the board, including the prebuilt llama.cpp archive used in the next step.

On first run the agent asks for this device's identity and writes `device_profile.json` itself, so a fresh clone needs nothing here. If you have run the software on the host PC before, that clone already holds your host's `device_profile.json` (it is gitignored, not tracked): delete it from the copy before or after transferring, or the board comes up with the host's agent-ID and preset.

> To update the board later, re-run the same `scp`. If you would rather `git pull` on the board itself, install git there (`apt-get install git`) and clone directly instead.

## 5. llama.cpp (leader only)

Only needed if this board acts as **leader** and runs FunctionGemma. Skip it for a sensing-only board.

Extract the prebuilt Cortex-A35 build into site-packages. No pip and no compiler needed:

```sh
tar xzf /home/weston/IMPLEMENTATION/LeaderLogic/stm32mp257fdk/prebuilt/llama_cpp_python-0.3.23-cortexa35.tar.gz -C /usr/lib/python3.12/site-packages/
```

Install its two Python dependencies:

```sh
apt-get install python3-diskcache python3-jinja2
```

Tell the dynamic linker where the extracted `.so` files are, otherwise the import fails with `libggml.so.0: cannot open shared object file`:

```sh
echo /usr/lib/python3.12/site-packages/llama_cpp/lib > /etc/ld.so.conf.d/llama_cpp.conf
ldconfig
```

`ldconfig` warns that `libmtmd.so.0 is not a symbolic link`. That is harmless, it comes from the upstream wheel.

Check you got the optimized build:

```sh
python3 -c "from llama_cpp import llama_cpp; llama_cpp.llama_backend_init(); print(llama_cpp.llama_print_system_info().decode())"
# CPU : NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | REPACK = 1 |
```

Details, requirements, and how to rebuild it: [`prebuilt/README.md`](prebuilt/README.md).

### The LLM

You do not need to copy the model by hand. On first run the leader downloads `functiongemma-270m-it-Q4_K_M.gguf` from `unsloth/functiongemma-270m-it-GGUF` into `IMPLEMENTATION/models/` (253 MB) and reuses that copy afterwards.

It needs `huggingface-hub`, which comes from pip:

```sh
apt-get install python3-pip
python3 -m pip install huggingface-hub
```

Note there is no bare `pip` command on these images: use `pip3` or `python3 -m pip`.

If pip fails to run at all (a broken vendored `resolvelib` shows up as `ImportError: cannot import name 'RequirementInformation'`), download the `.gguf` on your PC and copy it over instead:

```sh
scp functiongemma-270m-it-Q4_K_M.gguf root@<board-ip>:/home/weston/IMPLEMENTATION/models/
```

## 6. Run it

Install the base dependencies first. Every device needs them, whatever its role, and on this image they come from apt rather than pip:

```sh
apt-get install python3-psutil python3-requests
```

Then:

```sh
cd /home/weston/IMPLEMENTATION
python3 main.py
```

Role-specific dependencies are listed in [`requirements.txt`](requirements.txt) in this folder and in the sensing preset's own file under [`SensingLogic/`](../../SensingLogic/). Full configuration reference: [`IMPLEMENTATION/README.md`](../../README.md).
