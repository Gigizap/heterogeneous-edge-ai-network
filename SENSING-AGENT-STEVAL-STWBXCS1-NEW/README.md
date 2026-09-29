# STWIN.box sensing agent

Two ST names appear in this folder: **STEVAL-STWINBX1** is the STWIN.box kit (order code), and **STEVAL-STWBXCS1** is the core system board inside it (STM32U585 microcontroller, EMW3080 Wi-Fi module, sensors), which runs this firmware. ST's files use the kit name (`STWINBX1`).

C firmware for the **STWIN.box** that runs as a **sensing agent** of the peer-to-peer network in [Gigizap/heterogeneous-edge-ai-network](https://github.com/Gigizap/heterogeneous-edge-ai-network). It uses the same discovery and JSON-RPC protocol as the Python agents, so a Python leader discovers the board automatically and calls its tools.

| Tool | Result |
|---|---|
| `read_temperature` | Board temperature: `29.8 C (board temperature)` |
| `read_magnetic_field` | Magnetic field per axis and total: `X 118 mG, Y -196 mG, Z -410 mG, total 469 mG` |

Default identity, set in the firmware:

| Setting | Value |
|---|---|
| Agent-ID | `micro-controller` |
| TCP port | `5555` |
| Leader preset | none: the board is always a sensing agent |
| Sensing preset | the two tools above |

**How to set up the microcontroller STWIN.box as a sensing agent: [SETUP_STWINBOX.md](SETUP_STWINBOX.md)** (install, build, flash and use, step by step).

---

## Required edits before building

All paths below are relative to `C:\ST\STWINBX1_WIFI`, the firmware assembled in step 2 of [SETUP_STWINBOX.md](SETUP_STWINBOX.md). `<APP>` = `Projects/STWIN.box/Applications/NetXDuo/Nx_WebServer`.

| # | When | File | Change |
|---|---|---|---|
| 1 | **Always** | `<APP>/Core/Inc/mx_wifi_conf.h` | `WIFI_SSID` and `WIFI_PASSWORD`: your network's name and password, inside the quotes, exact upper/lower case. The network must be 2.4 GHz, WPA2, without a login page. |
| 2 | **More than one board on the network** | `<APP>/NetXDuo/App/app_netxduo.c` | `AGENT_ID`: a different name for each board, e.g. `"micro-controller-2"`. Every device on the network must have a unique agent-ID: the other agents store peers and tool owners by ID, so two boards with the same ID overwrite each other and replies go to the wrong board. |
| 3 | The Python network uses another discovery port | `<APP>/NetXDuo/App/app_netxduo.c` | `DISCOVERY_PORT` (default `9999`, same as `ConnectionLogic/discovery.py`). All devices must use the same value. |
| 4 | Optional | `<APP>/NetXDuo/App/app_netxduo.c` | `AGENT_TCP_PORT` (default `5555`). Each board has its own IP address, so boards can share this value. |

After editing, rebuild and flash (steps 6-7 of [SETUP_STWINBOX.md](SETUP_STWINBOX.md)).

To change the tools: `TOOLS_JSON` (names and descriptions sent to the leader) and `run_tool()` (what each tool returns), both in `app_netxduo.c`.

---

## Contents of this folder

| Item | Description |
|---|---|
| `README.md` | This file. |
| `SETUP_STWINBOX.md` | How to set up the STWIN.box as a sensing agent: install, build, flash and use, step by step. |
| `log_receiver.py` | Prints the board's debug log (UDP port 9998). |
| `STWINbox_pressure_sensor_issue.txt` | Notes on the on-board ILPS22QS pressure sensor. |
| `modified_files/` | The four ST files changed for the sensing agent (`app_netxduo.c`, `app_netxduo.h`, `stm32u5xx_hal_conf.h`, `mx_wifi_conf.h`), at their paths inside ST's STWINBX1_WIFI package. |

The rest of the firmware (ST's STWINBX1_WIFI package and ST's sensor drivers) is downloaded at fixed versions in step 2 of [SETUP_STWINBOX.md](SETUP_STWINBOX.md), and the files of `modified_files/` are copied over it.

---

## Architecture

### Hardware

| Part | Role |
|---|---|
| STM32U585 (Cortex-M33, 160 MHz) | Runs all the firmware, including the TCP/IP stack. |
| EMW3080 | Wi-Fi radio, connected over SPI in bypass mode: it carries network frames between the air and the STM32. |
| STTS22H (I2C2, 0x3F) | Temperature sensor, on the circuit board. |
| IIS2MDC (I2C2, 0x1E) | 3-axis magnetometer. |

### Software layers

| Layer | Location | Role |
|---|---|---|
| HAL | `Drivers/STM32U5xx_HAL_Driver` | Chip peripherals: clocks, GPIO, I2C, SPI. |
| BSP | `Drivers/BSP/STWIN.box` | Board wiring: power rails, LEDs, the I2C2 sensor bus. |
| Wi-Fi driver | `Drivers/BSP/Components/mx_wifi` | EMW3080 over SPI. |
| ThreadX | `Middlewares/ST/threadx` | Real-time OS: threads, sleeps, mutexes. |
| NetX Duo | `Middlewares/ST/netxduo` | TCP/IP stack: IP, DHCP, UDP, TCP. |
| Sensor drivers | `stts22h_reg.c`, `iis2mdc_reg.c`, `hts221_reg.c` | ST register-level drivers, one per sensor. |
| Application | `<APP>/NetXDuo/App/app_netxduo.c` | Sensors, discovery, TCP server, tools. |

---

## What the firmware does

Everything below is in `<APP>/NetXDuo/App/app_netxduo.c`.

### Startup

1. `main.c` configures clocks and power rails, then starts ThreadX.
2. `MX_NetXDuo_Init()` creates the packet pool, the IP instance (with the EMW3080 driver), enables ARP, ICMP, UDP and TCP, creates the DHCP client and the **main thread**.
3. `App_Main_Thread_Entry()` (main thread):
   1. starts DHCP and waits for an IP address;
   2. opens the log socket (`udp_log()`);
   3. powers the sensors and starts the I2C2 bus (`sensors_bus_init()`), logs every chip found on the bus (`i2c_scan()`), and sets up each sensor (`temperature_init()`, `humidity_init()`, `magnetic_init()`); each returns 1 only if the chip answers with its expected ID;
   4. waits 1.5 s for the first measurements and logs one line with all values;
   5. opens the discovery socket (UDP 9999) and starts the **agent thread**;
   6. runs the discovery loop forever.

### Main thread: discovery

Each pass of the loop (about every 100 ms):

| Step | Function | What it does |
|---|---|---|
| Announce | `send_hello()` | Every 2 s, broadcasts `{"type": "HELLO", "id": "micro-controller", "port": 5555}` to UDP 9999. |
| Listen | `discovery_receive()` → `handle_hello()` | Reads HELLOs from other devices and stores sender ID, IP and port (`peer_register()`). Logs `peer found`. |
| Expire | `peers_expire()` | Removes devices not heard for 15 s. Logs `peer lost`. |
| Status | `log_printf()` | Every 30 s logs `alive, N peer(s)`. |

The peer table holds up to 8 devices and is protected by a mutex, because the agent thread reads it to address replies.

### Agent thread: requests and replies

`Agent_Thread_Entry()` runs a TCP server on port 5555. For each connection:

1. `tcp_receive_line()` reads one JSON message (up to the newline or until the sender closes).
2. The connection is closed and the server listens again.
3. `handle_request()` decides what to do:

| Message | Action |
|---|---|
| has a `"type"` field (election, backup, capability messages) | Logged and ignored: these belong to leader-capable devices. |
| `tools/list` | Replies with `TOOLS_JSON`, the two tool descriptions. |
| `tools/call` | `run_tool()` reads the sensor and replies with the text result. |
| any other `method` | Replies with error `-32601 method not found`. |
| no `id` | JSON-RPC notification: no reply. |

4. `send_to_peer()` sends the reply as one JSON line on a **new** TCP connection to the sender. The sender is found by its `"from"` ID in the peer table, so a reply needs the sender's HELLO first. This is the same as the Python `P2PTransport.send()`.

`run_tool()` produces:

| Tool | Reads | Reply text |
|---|---|---|
| `read_temperature` | `temperature_read_tenths()`: STTS22H raw value (1/100 °C) | `29.8 C (board temperature)` |
| `read_magnetic_field` | `magnetic_read_mgauss()`: IIS2MDC raw X/Y/Z × 1.5 mG | `X .. mG, Y .. mG, Z .. mG, total .. mG` |

Values are kept as integers (tenths of a degree, milligauss) and printed without floating-point `printf`. After startup only the agent thread reads the sensors, so the I2C bus has a single user.

### Helper functions

| Function | Purpose |
|---|---|
| `json_find()`, `json_get_string()`, `json_get_raw()` | Read fields from the flat JSON messages. `json_get_raw()` copies the request `id` exactly as written, so the reply echoes it with the same type. |
| `sensor_read()`, `sensor_write()` | One I2C2 read/write pair for all sensors; each driver context carries its sensor's I2C address. |
| `format_tenths()` | `274` → `"27.4"`. |
| `log_printf()`, `udp_log()` | Debug log lines to UDP port 9998. |

---

## Protocol

Same message shapes as the Python reference (`ConnectionLogic/`, `SensingLogic/sensing_agent.py`, `LeaderLogic/tool_dispatcher.py`).

| Direction | Transport | Message |
|---|---|---|
| Board → all | UDP 9999, every 2 s | `{"type": "HELLO", "id": "micro-controller", "port": 5555}` |
| Leader → board | TCP 5555 | `{"jsonrpc": "2.0", "id": "list-1", "method": "tools/list", "from": "laptop-leader"}` |
| Board → leader | TCP, leader's port | `{"jsonrpc": "2.0", "id": "list-1", "result": {"tools": [...]}, "from": "micro-controller"}` |
| Leader → board | TCP 5555 | `{"jsonrpc": "2.0", "id": "rpc-7", "method": "tools/call", "params": {"name": "read_temperature", "arguments": {}}, "from": "laptop-leader"}` |
| Board → leader | TCP, leader's port | `{"jsonrpc": "2.0", "id": "rpc-7", "result": {"content": [{"type": "text", "text": "29.8 C (board temperature)"}]}, "from": "micro-controller"}` |

Errors use the codes and messages of `sensing_agent.py`:

| Code | Message |
|---|---|
| `-32601` | `unknown tool: <name>` / `method not found: <method>` |
| `-32603` | `internal error in '<tool>': sensor not found` |

Tool descriptions (`TOOLS_JSON`) use the `tool_defs` format of the Python `tool_config.json` files.

---

## Debug log (over Wi-Fi, no debugger needed)

The STWIN.box has no built-in debugger: its `printf` serial console is only readable through an external ST-LINK debugger/programmer probe (e.g. STLINK-V3MINIE). Instead, this firmware sends its log **over Wi-Fi**: every event is broadcast as one text line to UDP port 9998, and `log_receiver.py` on any PC of the same network prints them. USB (for flashing) and Wi-Fi are all that is needed.

Run it on the PC (step 8 of [SETUP_STWINBOX.md](SETUP_STWINBOX.md)):

```text
python C:\ST\SENSING-AGENT-STEVAL-STWBXCS1\log_receiver.py
```

Each line starts with the board's IP address. Example output:

```text
I2C2 devices: 0x1E 0x2D 0x3F 0x53 0x57 0x5C
Sensors: I2C2 OK, temperature 29.9, humidity NOT found, magnetic OK (118 -196 -410 mG)
Agent micro-controller at 192.168.137.83: TCP 5555, discovery UDP 9999, no leader preset
peer found: laptop-leader 192.168.137.1:5556
tools/list from laptop-leader -> 2 tools
tools/call read_temperature from laptop-leader -> 29.8 C (board temperature)
alive, 2 peer(s)
```

---

## Project layout

`STWINBX1_WIFI` uses ST's STM32Cube package layout: shared code (`Drivers`, `Middlewares`) once at the top, projects under `Projects/<board>/Applications/<middleware>/<name>`. The IDE project in `<APP>/STM32CubeIDE/` reaches the shared code through relative paths, so `<APP>` stays inside `STWINBX1_WIFI`.

Place `STWINBX1_WIFI` at a short path, e.g. `C:\ST\STWINBX1_WIFI`: Windows limits a file path to 260 characters by default ([MAX_PATH](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)), and the deepest file is about 160 characters below that folder, so its own path must stay under about 95 characters.

| Folder in `<APP>/` | Content |
|---|---|
| `Core/Src` | `main.c`: clocks, power, ThreadX start. |
| `Core/Inc` | Configuration: `mx_wifi_conf.h` (Wi-Fi name and password), `stm32u5xx_hal_conf.h` (enabled HAL modules), `tx_user.h` (ThreadX), and the sensor driver headers. |
| `NetXDuo/App` | `app_netxduo.c/.h`: the application. |
| `AZURE_RTOS/App` | ThreadX start-up and memory pools. |
| `STM32CubeIDE/` | IDE project files and linker script. Building creates `STM32CubeIDE/Debug/Nx_WebServer.elf`, the file to flash. |
| `STM32CubeIDE/Application/User/Core/` | Sensor driver sources and `STWIN.box_bus.c`. This folder is compiled as a whole, so new `.c` files go here. |

The project keeps the name of the ST example it is built on, `Nx_WebServer`. Code is edited directly in the files above; `Nx_WebServer.ioc` (STM32CubeMX) is not used.

---

## Notes on the sensing agent

- **Replies need the leader's HELLO.** A reply is addressed by the sender's ID, so the board answers once it has heard that device's HELLO (at most 2 s after the leader starts). The Python leader retries `tools/list` for this reason.
- **Board temperature.** The STTS22H is on the circuit board and reads a few degrees above room temperature.
- **Announcements** go to `255.255.255.255`; Python peers receive them on UDP 9999 like any HELLO.
- **Peer table**: up to 8 devices at a time.
- **JSON reader**: handles the flat messages of this protocol; IDs and names are plain text without escaped characters.
- **Wi-Fi at boot**: the board joins the network present at power-on; RESET reconnects it.

---

## Credits

Each file keeps its original license header (see also `LICENSE.md` in `STWINBX1_WIFI`).

**[stm32-hotspot/STWINBX1_WIFI](https://github.com/stm32-hotspot/STWINBX1_WIFI)** (ST), commit `024b03b` (downloaded in step 2.2): the base of the firmware: the `Nx_WebServer` example project, HAL, BSP, ThreadX, NetX Duo, the EMW3080 driver and the Wi-Fi module firmware updater. Files changed for the sensing agent (in `modified_files/`):
- `<APP>/NetXDuo/App/app_netxduo.c`: ST's network and thread setup, with the application added (UDP log, sensors, sensing agent).
- `<APP>/NetXDuo/App/app_netxduo.h`: ST's file, adapted to the application.
- `<APP>/Core/Inc/stm32u5xx_hal_conf.h`: ST's file with the I2C module enabled.
- `<APP>/Core/Inc/mx_wifi_conf.h`: ST's file with placeholder Wi-Fi name and password.

`<APP>/STM32CubeIDE/Application/User/Core/STWIN.box_bus.c` is ST's file copied unchanged from `Drivers/BSP/STWIN.box/` in step 2.4 (I2C2 bus setup).

**[STMicroelectronics/stts22h-pid](https://github.com/STMicroelectronics/stts22h-pid)**: `stts22h_reg.c` / `stts22h_reg.h`, downloaded unchanged in step 2.3 (commit `ae9a904`). Temperature sensor driver.

**[STMicroelectronics/iis2mdc-pid](https://github.com/STMicroelectronics/iis2mdc-pid)**: `iis2mdc_reg.c` / `iis2mdc_reg.h`, downloaded unchanged in step 2.3 (commit `b47c6d9`). Magnetometer driver.

**[STMicroelectronics/hts221-pid](https://github.com/STMicroelectronics/hts221-pid)**: `hts221_reg.c` / `hts221_reg.h`, downloaded unchanged in step 2.3 (commit `5c5e7c0`). Humidity sensor driver, used by `humidity_init()` at startup.

**[STMicroelectronics/STMems_Standard_C_drivers](https://github.com/STMicroelectronics/STMems_Standard_C_drivers)**: reference for the sensor setup sequences (ID check, reset, data rate, raw read), following its examples `stts22h_read_data_polling.c`, `iis2mdc_read_data_polling.c` and `hts221_read_data_polling.c`.

**[Gigizap/heterogeneous-edge-ai-network](https://github.com/Gigizap/heterogeneous-edge-ai-network)**: reference for the network protocol, re-implemented in C from `ConnectionLogic/discovery.py`, `ConnectionLogic/transport.py` and `SensingLogic/sensing_agent.py`; tool descriptions follow `SensingLogic/*/tool_config.json`; reply formats follow `LeaderLogic/tool_dispatcher.py`.

---

## License

- `README.md`, `SETUP_STWINBOX.md`, `log_receiver.py` and `STWINbox_pressure_sensor_issue.txt`: CC BY-NC-SA 4.0, like the rest of this repository.
- `modified_files/`: files of STMicroelectronics modified for the sensing agent. They stay under ST's license (SLA0044, text in `STWINBX1_WIFI/Projects/LICENSE.md`); the modifications are Copyright (c) 2026 Gigizap, as noted in each file's header. They are not covered by CC BY-NC-SA 4.0.
- The files downloaded in step 2 are not part of this repository and keep their own licenses (listed in `STWINBX1_WIFI/LICENSE.md` and in each driver repository).
