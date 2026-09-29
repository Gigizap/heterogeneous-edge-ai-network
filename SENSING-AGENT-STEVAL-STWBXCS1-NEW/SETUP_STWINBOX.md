# STWIN.box - sensing agent (temperature + magnetic field over Wi-Fi) - steps

How it works and credits: [README.md](README.md).

In these steps the folder containing this file is placed at `C:\ST\SENSING-AGENT-STEVAL-STWBXCS1`, and the firmware is assembled in `C:\ST\STWINBX1_WIFI`.

## 0. What you need

- STWIN.box (STEVAL-STWINBX1) and a USB-C data cable. No microSD card, no ST-LINK debugger.
- A Windows PC.
- A 2.4 GHz Wi-Fi network (the module does not support 5 GHz), normal password (WPA2), no login page.
  A Windows mobile hotspot works. Phone hotspots often block traffic between devices (client isolation): avoid them.
- The PC connected to the same network.

## 1. Download and install the programs

No ST account needed: on each st.com download page click "Download as a guest", enter name, surname and email, and you receive the download link by email.

- **1.1** STM32CubeProgrammer (Windows), from st.com. Install with default settings.
- **1.2** STM32CubeIDE, from st.com: the normal Windows installer (NOT "STM32CubeIDE for Visual Studio Code"). Tested with version 2.2.0, installed in `C:\ST\STM32CubeIDE_2.2.0`.
- **1.3** Python 3, from python.org. During install tick "Add python.exe to PATH". Needed for log_receiver.py and for the Python agents of step 9.
- **1.4** STM32CubeMX is NOT needed.

## 2. Get the files

The folder `SENSING-AGENT-STEVAL-STWBXCS1` (from the repository) holds only the files modified for the sensing agent: the 4 ST files in its `modified_files` folder.
All the other code is ST's and is downloaded from ST, at fixed versions.
At the end of step 2 the finished code is in `C:\ST\STWINBX1_WIFI`: ST's code + the 4 modified files. Steps 3-7 use that folder.

- **2.1** The folder `SENSING-AGENT-STEVAL-STWBXCS1`: on the repository's GitHub page click the green "Code" button > "Download ZIP". Right-click the zip > "Extract All...", click Extract.
  From the extracted repository, copy the folder `SENSING-AGENT-STEVAL-STWBXCS1` into `C:\ST`, so that you get `C:\ST\SENSING-AGENT-STEVAL-STWBXCS1\README.md`.
- **2.2** ST's STWINBX1_WIFI package (version 024b03b of 21 Dec 2023): open this link, the download starts:
  https://github.com/stm32-hotspot/STWINBX1_WIFI/archive/024b03b0fd7645c8e42148e58bf8826b7401530b.zip
  Right-click the zip > "Extract All...", and as destination type `C:\ST`, then click Extract.
  Rename the folder `C:\ST\STWINBX1_WIFI-024b03b0fd7645c8e42148e58bf8826b7401530b` to `STWINBX1_WIFI`, so that you get `C:\ST\STWINBX1_WIFI\README.md`.
  Use a short folder like `C:\ST`. By default Windows limits a file path to 260 characters, and the deepest file
  in STWINBX1_WIFI sits about 160 characters below it, so that folder's own path must be shorter than about 95 characters.
  If it is too long, Windows Explorer's "Extract All" shows errors or skips files.
- **2.3** ST's sensor drivers: right-click each link > "Save link as...", and save the file with the same name into the folder shown.
  Into `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\Core\Inc`:
  - https://raw.githubusercontent.com/STMicroelectronics/stts22h-pid/ae9a9042b888cdfcc4cd5315711df8ef0e6819f2/stts22h_reg.h
  - https://raw.githubusercontent.com/STMicroelectronics/iis2mdc-pid/b47c6d98c17cd77f76f7ca6e407cc82af8afea3a/iis2mdc_reg.h
  - https://raw.githubusercontent.com/STMicroelectronics/hts221-pid/5c5e7c07cdebeeb49c07c82c10e462dc3e0c7ce8/hts221_reg.h

  Into `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\STM32CubeIDE\Application\User\Core`:
  - https://raw.githubusercontent.com/STMicroelectronics/stts22h-pid/ae9a9042b888cdfcc4cd5315711df8ef0e6819f2/stts22h_reg.c
  - https://raw.githubusercontent.com/STMicroelectronics/iis2mdc-pid/b47c6d98c17cd77f76f7ca6e407cc82af8afea3a/iis2mdc_reg.c
  - https://raw.githubusercontent.com/STMicroelectronics/hts221-pid/5c5e7c07cdebeeb49c07c82c10e462dc3e0c7ce8/hts221_reg.c
- **2.4** Copy the file `C:\ST\STWINBX1_WIFI\Drivers\BSP\STWIN.box\STWIN.box_bus.c` into `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\STM32CubeIDE\Application\User\Core` (the same folder as the three `.c` files of 2.3).
- **2.5** Put the 4 modified files into the finished code. `modified_files` has the same folders as `STWINBX1_WIFI`, so one copy puts each file in its place:
  open `C:\ST\SENSING-AGENT-STEVAL-STWBXCS1\modified_files`: it contains one folder, `Projects`. Click `Projects`, Ctrl+C.
  Open `C:\ST\STWINBX1_WIFI`, Ctrl+V. Windows asks about files that already exist: click "Replace the files in the destination".
  The 4 files replaced are `app_netxduo.c`, `app_netxduo.h`, `stm32u5xx_hal_conf.h` and `mx_wifi_conf.h`.
- **2.6** Check: `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\STM32CubeIDE\Application\User\Core` now contains
  `STWIN.box_bus.c`, `hts221_reg.c`, `iis2mdc_reg.c`, `stts22h_reg.c`, `syscalls.c`, `sysmem.c`.
- **2.7** Do not move folders inside `STWINBX1_WIFI`: the project finds its files by relative paths.

## 3. Update the Wi-Fi module firmware (only once per board)

- **3.1** Connect the board to the PC with the USB-C cable.
- **3.2** DFU mode: hold RESET, press USER, release RESET, release USER.
- **3.3** Open STM32CubeProgrammer. Top right: select USB, click refresh next to Port, click Connect.
- **3.4** Left sidebar: Erasing & Programming. File path: Browse > `C:\ST\STWINBX1_WIFI\WiFi_module_upgrade\EMW3080update v2.3.4 - STWINBX1.hex`
- **3.5** Tick "Run after programming", click Start Programming.
- **3.6** Warning "Connection to device 0x482 is lost": click OK. "Start operation achieved successfully": click OK.
  (Normal: the board restarted with the updater. Green and orange LEDs turn on.)
- **3.7** Press the USER button once (short press). The orange LED turns off for a few seconds, then comes back on and later blinks.
- **3.8** Wait 2-3 minutes without unplugging the board.

## 4. Import the project (STM32CubeIDE)

- **4.1** Open STM32CubeIDE, workspace `C:\ST\workspace`, Launch. Close the "Information Center" tab if it opens.
- **4.2** Menu File > Import... > General > Existing Projects into Workspace > Next.
- **4.3** Select root directory: Browse > `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\STM32CubeIDE` > Select Folder.
- **4.4** Nx_WebServer ticked, "Copy projects into workspace" NOT ticked. Click Finish (accept if asked to upgrade the project).
  The project is still called Nx_WebServer (ST's example it started from); it no longer contains a web server.
- **4.5** Do NOT open the `Nx_WebServer.ioc` file (STM32CubeMX would overwrite the code).

## 5. Set your Wi-Fi name and password

- **5.1** Menu File > Open File... > `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\Core\Inc\mx_wifi_conf.h`
- **5.2** Ctrl+F, search `WIFI_SSID`.
- **5.3** Change the two lines, keeping the quotes (upper/lower case must match):
  ```c
  #define WIFI_SSID        "YourNetworkName"
  #define WIFI_PASSWORD    "YourPassword"
  ```
- **5.4** Ctrl+S to save.
- **5.5** Only if you use more than one board on the same network: Menu File > Open File... > `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\NetXDuo\App\app_netxduo.c`,
  Ctrl+F, search `AGENT_ID`, and give each board a different name, keeping the quotes (e.g. `"micro-controller-2"`). Ctrl+S to save.

## 6. Build

- **6.1** Click Nx_WebServer in Project Explorer, then the hammer icon (Build).
- **6.2** Console tab: "Build Finished. 0 errors". The warnings are normal: they are in ST's files (`nx_driver_emw3080.c`, `stm32u5xx_hal_dma_ex.c`) plus "LOAD segment with RWX permissions" from the linker.
- **6.3** The file to load is `C:\ST\STWINBX1_WIFI\Projects\STWIN.box\Applications\NetXDuo\Nx_WebServer\STM32CubeIDE\Debug\Nx_WebServer.elf`

## 7. Load the program on the board (STM32CubeProgrammer)

- **7.1** DFU mode: hold RESET, press USER, release RESET, release USER.
- **7.2** Top right: USB > refresh > Connect (if already connected from before: Disconnect first, then Connect).
- **7.3** Erasing & Programming > Browse > `Nx_WebServer.elf` (path in 6.3).
- **7.4** Tick "Run after programming" > Start Programming.
- **7.5** Warning "Connection to device 0x482 is lost": click OK (normal, the board restarted with the new program).

## 8. Read the board's log on the PC

The board sends its log over Wi-Fi (UDP port 9998), so no ST-LINK debugger is needed: log_receiver.py prints it on the PC.

- **8.1** The Wi-Fi network must be on before the board starts. If it was not, press RESET on the board.
- **8.2** Open cmd and type:
  ```
  python C:\ST\SENSING-AGENT-STEVAL-STWBXCS1\log_receiver.py
  ```
- **8.3** If Windows Firewall asks about Python, click Allow access.
- **8.4** Within about 30 seconds (Wi-Fi connection time) you see, with your board's IP in front:
  ```
  I2C2 devices: 0x1E 0x2D 0x3F 0x53 0x57 0x5C
  Sensors: I2C2 OK, temperature 29.9, humidity NOT found, magnetic OK (118 -196 -410 mG)
  Agent micro-controller at 192.168.137.83: TCP 5555, discovery UDP 9999, no leader preset
  ```
  and then "alive, N peer(s)" every 30 seconds.
- **8.5** Leave this window open: it shows everything the agent does.

## 9. Use it with the network

- **9.1** Start the Python agents of https://github.com/Gigizap/heterogeneous-edge-ai-network (`python main.py`) on a device of the same network, with a leader preset.
- **9.2** Nothing to configure on the board: the leader finds "micro-controller" by its HELLO, asks for its tools, and can call them from Telegram
  (e.g. "what's the temperature?"). The log window (step 8) shows the leader as a peer ("peer found: ...") and one line per request.
- **9.3** `/capabilities` in Telegram lists read_temperature and read_magnetic_field with owner micro-controller.
- **9.4** Bring a magnet or a phone close to the board and ask for the magnetic field again: the values change.

## If something does not work

- **No log lines at all:** Wi-Fi problem. Check SSID/password (step 5), that the network is 2.4 GHz and was on at boot, then press RESET.
- **Log OK but the leader does not list the board's tools:**
  - the log shows "cannot reply to <leader>: unknown peer": the board did not receive the leader's HELLO.
    On the leader's device allow Python through the firewall (inbound UDP 9999 and the leader's TCP port; the repo's `ConnectionLogic/README.md` has the exact commands);
  - the log shows no "peer found" for the leader: the devices cannot reach each other; check client isolation on the network.
- **Temperature reads about 8-10 C above the room:** normal, it measures the board.
- **Humidity NOT found:** normal, this board has no humidity sensor (the package is wrong).
- **Pressure is not offered:** dropped on purpose, see `STWINbox_pressure_sensor_issue.txt`.
