/* USER CODE BEGIN Header */
/**
  ******************************************************************************
* @file    app_netxduo.c
* @author  MCD Application Team
* @brief   NetXDuo applicative file
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* Modified for the STWIN.box sensing agent (github.com/Gigizap/heterogeneous-edge-ai-network):
 * UDP debug log, sensor reading and the sensing agent (discovery, TCP server, tools).
 * Modifications Copyright (c) 2026 Gigizap. This file remains under STMicroelectronics'
 * license (SLA0044, see the LICENSE file referenced above). */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "app_netxduo.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include   "app_azure_rtos.h"
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include "stts22h_reg.h"
#include "iis2mdc_reg.h"
#include "hts221_reg.h"
#include "STWIN.box.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* Define the ThreadX , NetX and FileX object control blocks. */

/* Define Threadx global data structures. */
TX_THREAD AppMainThread;
TX_SEMAPHORE Semaphore;

/* Define NetX global data structures. */

NX_PACKET_POOL AppPool;

NX_IP   IpInstance;
NX_DHCP DHCPClient;

ULONG IpAddress;
ULONG NetMask;

/* App memory pointer. */
UCHAR   *pointer;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* Main application thread: network setup, sensors, UDP logs */
static void  App_Main_Thread_Entry(ULONG thread_input);

/* DHCP state change notify callback */
static VOID ip_address_change_notify_callback(NX_IP *ip_instance, VOID *ptr);

/* USER CODE END PFP */
/**
  * @brief  Application NetXDuo Initialization.
  * @param memory_ptr: memory pointer
  * @retval int
  */
UINT MX_NetXDuo_Init(VOID *memory_ptr)
{
  UINT ret = NX_SUCCESS;
  TX_BYTE_POOL *byte_pool = (TX_BYTE_POOL*)memory_ptr;

   /* USER CODE BEGIN App_NetXDuo_MEM_POOL */

  /* USER CODE END App_NetXDuo_MEM_POOL */

  /* USER CODE BEGIN MX_NetXDuo_Init */
#if (USE_MEMORY_POOL_ALLOCATION == 1)  
  printf("Nx_WebServer_Application_Started..\n");
  
  /* Initialize the NetX system. */
  nx_system_initialize();

  /* Allocate the memory for packet_pool.  */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer,  NX_PACKET_POOL_SIZE, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }  
  
  /* Create the Packet pool to be used for packet allocation */
  ret = nx_packet_pool_create(&AppPool, "Main Packet Pool", PAYLOAD_SIZE, pointer, NX_PACKET_POOL_SIZE);
  
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }
  
  /* Allocate the memory for Ip_Instance */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer,   2 * DEFAULT_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }
  
  /* Create the main NX_IP instance */
  ret = nx_ip_create(&IpInstance, "Main Ip instance", NULL_ADDRESS, NULL_ADDRESS, &AppPool, nx_driver_emw3080_entry,
                     pointer, 2 * DEFAULT_MEMORY_SIZE, DEFAULT_PRIORITY);
  
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }
  
  /* Allocate the memory for ARP */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer, ARP_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }
  
  /* Enable the ARP protocol and provide the ARP cache size for the IP instance */
  ret = nx_arp_enable(&IpInstance, (VOID *)pointer, ARP_MEMORY_SIZE);
  
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  } 
  
  /* Enable the ICMP */
  ret = nx_icmp_enable(&IpInstance);
  
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }
  
  /* Enable the UDP protocol required for  DHCP communication */
  ret = nx_udp_enable(&IpInstance);
  
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }  
  
  /* Enable the TCP protocol */
  ret = nx_tcp_enable(&IpInstance);
  
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }
  
  /* Allocate the main thread. */
  ret = tx_byte_allocate(byte_pool, (VOID **) &pointer, 2 * DEFAULT_MEMORY_SIZE, TX_NO_WAIT);
  
  /* Check main thread memory allocation. */
  if (ret != NX_SUCCESS)
  {
    printf("Main thread memory allocation failed : 0x%02x\n", ret);
    Error_Handler();
  }
  
  /* Create the main thread */
  ret = tx_thread_create(&AppMainThread, "App Main thread", App_Main_Thread_Entry, 0, pointer, 2 * DEFAULT_MEMORY_SIZE,
                         DEFAULT_MAIN_PRIORITY, DEFAULT_MAIN_PRIORITY, TX_NO_TIME_SLICE, TX_AUTO_START);
  
  if (ret != TX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }
  
  /* create the DHCP client */
  ret = nx_dhcp_create(&DHCPClient, &IpInstance, "DHCP Client");
  
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }
  
  /* set DHCP notification callback  */
  tx_semaphore_create(&Semaphore, "App Semaphore", 0);

#endif
  /* USER CODE END MX_NetXDuo_Init */

  return ret;
}

/* USER CODE BEGIN 1 */
static NX_UDP_SOCKET LogSocket;

/* ---- On-board sensors, all on the I2C2 bus --------------------------------
 * Setup follows ST's official examples (github.com/STMicroelectronics/
 * STMems_Standard_C_drivers, <sensor>_STdC/examples). I2C addresses are the
 * ones used by ST's board support code (STWIN.box_env/motion_sensors.c).
 * Each driver context keeps its sensor's I2C address in .handle, so one pair
 * of read/write functions serves all sensors.
 * Values are returned as integers (tenths, or mG) because float printf is
 * not enabled in this project.
 */
static stmdev_ctx_t   stts22h_ctx;
static stts22h_priv_t stts22h_priv;
static stmdev_ctx_t   iis2mdc_ctx;
static stmdev_ctx_t   hts221_ctx;
static float_t        hum_x0, hum_y0, hum_x1, hum_y1;  /* HTS221 factory calibration points */

static int32_t sensor_write(void *handle, uint8_t reg, const uint8_t *bufp, uint16_t len)
{
  return BSP_I2C2_WriteReg((uint16_t)(uint32_t)handle, reg, (uint8_t *)bufp, len);
}

static int32_t sensor_read(void *handle, uint8_t reg, uint8_t *bufp, uint16_t len)
{
  return BSP_I2C2_ReadReg((uint16_t)(uint32_t)handle, reg, bufp, len);
}

static void sensor_delay(uint32_t ms)
{
  tx_thread_sleep((ms * TX_TIMER_TICKS_PER_SECOND) / 1000U + 1U);
}

static void sensor_ctx_init(stmdev_ctx_t *ctx, uint16_t address)
{
  ctx->write_reg = sensor_write;
  ctx->read_reg  = sensor_read;
  ctx->mdelay    = sensor_delay;
  ctx->handle    = (void *)(uint32_t)address;
  ctx->priv_data = NULL;
}

/* Power and bus for all sensors. Returns 1 if OK. */
static UINT sensors_bus_init(void)
{
  BSP_Enable_LDO();                       /* sensor supply, as in ST's factory firmware */
  return BSP_I2C2_Init() == BSP_ERROR_NONE;
}

/* Log the 7-bit address of every chip that answers on I2C2 (one-time check). */
static void i2c_scan(CHAR *msg, UINT size)
{
  snprintf(msg, size, "I2C2 devices:");
  for (UINT addr = 0x08; addr < 0x78; addr++)
  {
    if (BSP_I2C2_IsReady((uint16_t)(addr << 1), 1) == BSP_ERROR_NONE)
    {
      UINT len = strlen(msg);
      snprintf(msg + len, size - len, " 0x%02X", addr);
    }
  }
  strncat(msg, "\n", size - strlen(msg) - 1);
}

/* Each *_init() returns 1 if the sensor answered and is set up, 0 if not. */

static UINT temperature_init(void)   /* STTS22H */
{
  uint8_t whoami = 0;

  sensor_ctx_init(&stts22h_ctx, STTS22H_I2C_ADD_L);
  stts22h_ctx.priv_data = &stts22h_priv;
  stts22h_dev_id_get(&stts22h_ctx, &whoami);
  if (whoami != STTS22H_ID)
  {
    return 0;
  }
  stts22h_auto_increment_set(&stts22h_ctx, 1);
  stts22h_temp_data_rate_set(&stts22h_ctx, STTS22H_1Hz);
  return 1;
}

static UINT humidity_init(void)      /* HTS221 */
{
  uint8_t whoami = 0;

  sensor_ctx_init(&hts221_ctx, HTS221_I2C_ADDRESS);
  hts221_device_id_get(&hts221_ctx, &whoami);
  if (whoami != HTS221_ID)
  {
    return 0;
  }
  hts221_hum_adc_point_0_get(&hts221_ctx, &hum_x0);
  hts221_hum_rh_point_0_get(&hts221_ctx, &hum_y0);
  hts221_hum_adc_point_1_get(&hts221_ctx, &hum_x1);
  hts221_hum_rh_point_1_get(&hts221_ctx, &hum_y1);
  hts221_block_data_update_set(&hts221_ctx, PROPERTY_ENABLE);
  hts221_data_rate_set(&hts221_ctx, HTS221_ODR_1Hz);
  hts221_power_on_set(&hts221_ctx, PROPERTY_ENABLE);
  return 1;
}

static UINT magnetic_init(void)      /* IIS2MDC */
{
  uint8_t whoami = 0;
  uint8_t rst = 0;
  UINT tries = 0;

  sensor_ctx_init(&iis2mdc_ctx, IIS2MDC_I2C_ADD);
  iis2mdc_device_id_get(&iis2mdc_ctx, &whoami);
  if (whoami != IIS2MDC_ID)
  {
    return 0;
  }
  iis2mdc_reset_set(&iis2mdc_ctx, PROPERTY_ENABLE);
  do
  {
    iis2mdc_reset_get(&iis2mdc_ctx, &rst);
  } while (rst && ++tries < 10);
  iis2mdc_block_data_update_set(&iis2mdc_ctx, PROPERTY_ENABLE);
  iis2mdc_data_rate_set(&iis2mdc_ctx, IIS2MDC_ODR_10Hz);
  iis2mdc_set_rst_mode_set(&iis2mdc_ctx, IIS2MDC_SENS_OFF_CANC_EVERY_ODR);
  iis2mdc_offset_temp_comp_set(&iis2mdc_ctx, PROPERTY_ENABLE);
  iis2mdc_operating_mode_set(&iis2mdc_ctx, IIS2MDC_CONTINUOUS_MODE);
  return 1;
}

/* Temperature in tenths of a degree C (274 = 27.4 C). */
static INT temperature_read_tenths(void)
{
  int16_t raw = 0;                        /* sensor unit: 1/100 degree C */
  stts22h_temperature_raw_get(&stts22h_ctx, &raw);
  return (INT)raw / 10;
}

/* Relative humidity in tenths of % (453 = 45.3 %). */
static INT humidity_read_tenths(void)
{
  int16_t raw = 0;
  hts221_humidity_raw_get(&hts221_ctx, &raw);
  /* straight line through the two factory calibration points, as in ST's example */
  float_t rh = ((hum_y1 - hum_y0) * raw + (hum_x1 * hum_y0 - hum_x0 * hum_y1)) / (hum_x1 - hum_x0);
  if (rh < 0.0f)
  {
    rh = 0.0f;
  }
  if (rh > 100.0f)
  {
    rh = 100.0f;
  }
  return (INT)(rh * 10.0f + 0.5f);
}

/* Magnetic field X, Y, Z in milligauss. */
static void magnetic_read_mgauss(INT mg[3])
{
  int16_t raw[3] = {0};
  iis2mdc_magnetic_raw_get(&iis2mdc_ctx, raw);
  for (UINT i = 0; i < 3; i++)
  {
    mg[i] = (INT)raw[i] * 3 / 2;          /* 1.5 mG per LSB */
  }
}

/* Write a value given in tenths as text: 274 -> "27.4", -5 -> "-0.5". */
static void format_tenths(CHAR *out, UINT size, INT tenths)
{
  INT a = (tenths < 0) ? -tenths : tenths;
  snprintf(out, size, "%s%d.%d", (tenths < 0) ? "-" : "", a / 10, a % 10);
}

/* Send one text line to the PC (UDP broadcast, port 9998) */
static void udp_log(const char *text)
{
  NX_PACKET *packet;

  if (nx_packet_allocate(&AppPool, &packet, NX_UDP_PACKET, TX_WAIT_FOREVER) != NX_SUCCESS)
  {
    return;
  }
  if (nx_packet_data_append(packet, (VOID *)text, strlen(text), &AppPool, TX_WAIT_FOREVER) != NX_SUCCESS)
  {
    nx_packet_release(packet);
    return;
  }
  if (nx_udp_socket_send(&LogSocket, packet, IP_ADDRESS(255, 255, 255, 255), 9998) != NX_SUCCESS)
  {
    nx_packet_release(packet);
  }
}

/* printf-style log line to the PC (newline added). Safe to call from any thread. */
static void log_printf(const CHAR *format, ...)
{
  CHAR line[200];
  va_list args;

  va_start(args, format);
  vsnprintf(line, sizeof(line) - 1, format, args);
  va_end(args);
  strcat(line, "\n");
  udp_log(line);
}

static void ip_to_str(ULONG ip, CHAR *out, UINT size)
{
  snprintf(out, size, "%lu.%lu.%lu.%lu", (ip >> 24) & 0xFF, (ip >> 16) & 0xFF, (ip >> 8) & 0xFF, ip & 0xFF);
}

/* ---- Sensing agent ---------------------------------------------------------
 * Same protocol as the Python reference (github.com/Gigizap/heterogeneous-edge-ai-network,
 * IMPLEMENTATION/ConnectionLogic and SensingLogic/sensing_agent.py):
 *  - discovery: UDP HELLO {"type","id","port"} broadcast on port 9999 every 2 s;
 *    peers heard via HELLO are kept for 15 s;
 *  - transport: one newline-terminated JSON message per TCP connection; a reply
 *    goes to the sender's id (looked up in the peer table) on a new connection,
 *    with "from" set to our id;
 *  - no leader preset: like a Python device without one, this device never runs
 *    the election. It only answers JSON-RPC tools/list and tools/call; messages
 *    with a "type" (election, backup, capability) are ignored.
 * Identity and tools are hardcoded: there is no setup step on the board.
 */
#define AGENT_ID           "micro-controller"
#define AGENT_TCP_PORT     5555
#define DISCOVERY_PORT     9999
#define ANNOUNCE_INTERVAL  (2 * TX_TIMER_TICKS_PER_SECOND)
#define PEER_TIMEOUT       (15 * TX_TIMER_TICKS_PER_SECOND)
#define REPLY_TIMEOUT      (3 * TX_TIMER_TICKS_PER_SECOND)
#define MAX_PEERS          8
#define PEER_ID_SIZE       48

/* tools/list answer: same format as tool_defs in a Python tool_config.json */
static const CHAR TOOLS_JSON[] =
  "[{\"type\": \"function\", \"function\": {\"name\": \"read_temperature\", "
  "\"description\": \"reads the board temperature in degrees Celsius (a few degrees above room temperature). "
  "Use this when the user asks: 'what's the temperature?' or 'how hot is it?'.\", "
  "\"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}, "
  "{\"type\": \"function\", \"function\": {\"name\": \"read_magnetic_field\", "
  "\"description\": \"detects the magnetic field on the X, Y and Z axes in milligauss, with the total strength. "
  "Use this when the user asks: 'what's the magnetic field?' or 'is there a magnet nearby?'.\", "
  "\"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}]";

typedef struct
{
  CHAR  id[PEER_ID_SIZE];                 /* empty = free slot */
  ULONG ip;
  UINT  port;
  ULONG last_seen;                        /* tx_time_get() ticks */
} Peer;

static Peer          Peers[MAX_PEERS];
static TX_MUTEX      PeersMutex;          /* table is written by the main thread, read by the agent thread */
static NX_UDP_SOCKET DiscoverySocket;
static NX_TCP_SOCKET ServerSocket;
static NX_TCP_SOCKET ClientSocket;
static TX_THREAD     AgentThread;
static ULONG         AgentThreadStack[3 * DEFAULT_MEMORY_SIZE / sizeof(ULONG)];
static UINT          TempOk, MagOk;       /* set once at startup */

/* ---- Minimal JSON reading: enough for the flat messages of this protocol.
 * Strings are taken as-is (no escape handling): ids and method names have none. */

/* Pointer to the value of "key", or NULL. */
static const CHAR *json_find(const CHAR *json, const CHAR *key)
{
  CHAR pattern[24];
  const CHAR *p = json;

  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  while ((p = strstr(p, pattern)) != NULL)
  {
    p += strlen(pattern);
    while (*p == ' ')
    {
      p++;
    }
    if (*p != ':')
    {
      continue;                           /* same text inside a value, not a key */
    }
    p++;
    while (*p == ' ')
    {
      p++;
    }
    return p;
  }
  return NULL;
}

/* Copy a string value without quotes. Returns 1 if found. */
static UINT json_get_string(const CHAR *json, const CHAR *key, CHAR *out, UINT size)
{
  const CHAR *p = json_find(json, key);
  const CHAR *end;

  if (p == NULL || *p != '"')
  {
    return 0;
  }
  end = strchr(p + 1, '"');
  if (end == NULL || (UINT)(end - p - 1) >= size)
  {
    return 0;
  }
  memcpy(out, p + 1, end - p - 1);
  out[end - p - 1] = '\0';
  return 1;
}

/* Copy a value exactly as written ("rpc-7" with its quotes, or 7), to echo the JSON-RPC id.
 * Returns 0 if missing or null (a notification: no reply expected). */
static UINT json_get_raw(const CHAR *json, const CHAR *key, CHAR *out, UINT size)
{
  const CHAR *p = json_find(json, key);
  const CHAR *end;

  if (p == NULL)
  {
    return 0;
  }
  if (*p == '"')
  {
    end = strchr(p + 1, '"');
    if (end == NULL)
    {
      return 0;
    }
    end++;
  }
  else
  {
    end = p;
    while (*end != '\0' && *end != ',' && *end != '}' && *end != ' ')
    {
      end++;
    }
  }
  if (end == p || (UINT)(end - p) >= size)
  {
    return 0;
  }
  memcpy(out, p, end - p);
  out[end - p] = '\0';
  return strcmp(out, "null") != 0;
}

/* ---- Peer table (Discovery._peers in Python) ---- */

static void peer_register(const CHAR *id, ULONG ip, UINT port)
{
  Peer *slot = NULL;
  UINT is_new = 0;
  CHAR ip_text[16];

  tx_mutex_get(&PeersMutex, TX_WAIT_FOREVER);
  for (UINT i = 0; i < MAX_PEERS && slot == NULL; i++)
  {
    if (strcmp(Peers[i].id, id) == 0)
    {
      slot = &Peers[i];
    }
  }
  for (UINT i = 0; i < MAX_PEERS && slot == NULL; i++)
  {
    if (Peers[i].id[0] == '\0')
    {
      slot = &Peers[i];
      strcpy(slot->id, id);
      is_new = 1;
    }
  }
  if (slot != NULL)
  {
    slot->ip        = ip;
    slot->port      = port;
    slot->last_seen = tx_time_get();
  }
  tx_mutex_put(&PeersMutex);

  ip_to_str(ip, ip_text, sizeof(ip_text));
  if (slot == NULL)
  {
    log_printf("peer table full, ignoring %s", id);
  }
  else if (is_new)
  {
    log_printf("peer found: %s %s:%u", id, ip_text, port);
  }
}

/* Drop peers not heard for PEER_TIMEOUT (Discovery._watchdog_loop in Python). */
static void peers_expire(void)
{
  CHAR lost[PEER_ID_SIZE];

  do
  {
    lost[0] = '\0';
    tx_mutex_get(&PeersMutex, TX_WAIT_FOREVER);
    for (UINT i = 0; i < MAX_PEERS && lost[0] == '\0'; i++)
    {
      if (Peers[i].id[0] != '\0' && tx_time_get() - Peers[i].last_seen > PEER_TIMEOUT)
      {
        strcpy(lost, Peers[i].id);
        Peers[i].id[0] = '\0';
      }
    }
    tx_mutex_put(&PeersMutex);
    if (lost[0] != '\0')
    {
      log_printf("peer lost: %s", lost);
    }
  } while (lost[0] != '\0');
}

/* Returns 1 and fills ip/port if the peer is known. */
static UINT peer_lookup(const CHAR *id, ULONG *ip, UINT *port)
{
  UINT found = 0;

  tx_mutex_get(&PeersMutex, TX_WAIT_FOREVER);
  for (UINT i = 0; i < MAX_PEERS && !found; i++)
  {
    if (Peers[i].id[0] != '\0' && strcmp(Peers[i].id, id) == 0)
    {
      *ip    = Peers[i].ip;
      *port  = Peers[i].port;
      found  = 1;
    }
  }
  tx_mutex_put(&PeersMutex);
  return found;
}

static UINT peer_count(void)
{
  UINT count = 0;

  tx_mutex_get(&PeersMutex, TX_WAIT_FOREVER);
  for (UINT i = 0; i < MAX_PEERS; i++)
  {
    count += (Peers[i].id[0] != '\0');
  }
  tx_mutex_put(&PeersMutex);
  return count;
}

/* ---- Discovery (UDP port 9999) ---- */

static void send_hello(void)
{
  CHAR hello[80];
  NX_PACKET *packet;

  snprintf(hello, sizeof(hello), "{\"type\": \"HELLO\", \"id\": \"%s\", \"port\": %u}", AGENT_ID, AGENT_TCP_PORT);
  if (nx_packet_allocate(&AppPool, &packet, NX_UDP_PACKET, TX_TIMER_TICKS_PER_SECOND) != NX_SUCCESS)
  {
    return;
  }
  if (nx_packet_data_append(packet, hello, strlen(hello), &AppPool, TX_TIMER_TICKS_PER_SECOND) != NX_SUCCESS
      || nx_udp_socket_send(&DiscoverySocket, packet, IP_ADDRESS(255, 255, 255, 255), DISCOVERY_PORT) != NX_SUCCESS)
  {
    nx_packet_release(packet);
  }
}

/* Register the sender of a HELLO {"type": "HELLO", "id": ..., "port": ...}. */
static void handle_hello(const CHAR *json, ULONG ip)
{
  CHAR type[16];
  CHAR id[PEER_ID_SIZE];
  const CHAR *port = json_find(json, "port");

  if (!json_get_string(json, "type", type, sizeof(type)) || strcmp(type, "HELLO") != 0)
  {
    return;
  }
  if (!json_get_string(json, "id", id, sizeof(id)) || strcmp(id, AGENT_ID) == 0 || port == NULL)
  {
    return;                               /* malformed, or our own broadcast */
  }
  peer_register(id, ip, (UINT)strtoul(port, NULL, 10));
}

/* Wait up to `wait` ticks for HELLOs and handle all that arrived. */
static void discovery_receive(ULONG wait)
{
  static CHAR buffer[256];
  NX_PACKET *packet;

  while (nx_udp_socket_receive(&DiscoverySocket, &packet, wait) == NX_SUCCESS)
  {
    ULONG ip = 0;
    UINT src_port = 0;
    ULONG len = 0;

    wait = NX_NO_WAIT;                    /* then just drain what is queued */
    nx_udp_source_extract(packet, &ip, &src_port);
    if (packet->nx_packet_length < sizeof(buffer))
    {
      nx_packet_data_retrieve(packet, buffer, &len);
    }
    nx_packet_release(packet);
    buffer[len] = '\0';
    handle_hello(buffer, ip);
  }
}

/* ---- Tools ---- */

/* Run one tool. Returns 0 with the result text, or a JSON-RPC error code with the
 * error message (same codes and messages as sensing_agent.py). */
static INT run_tool(const CHAR *name, CHAR *text, UINT size)
{
  if (strcmp(name, "read_temperature") == 0)
  {
    CHAR value[16];

    if (!TempOk)
    {
      snprintf(text, size, "internal error in '%s': sensor not found", name);
      return -32603;
    }
    format_tenths(value, sizeof(value), temperature_read_tenths());
    snprintf(text, size, "%s C (board temperature)", value);
    return 0;
  }
  if (strcmp(name, "read_magnetic_field") == 0)
  {
    INT mg[3];
    float_t total;

    if (!MagOk)
    {
      snprintf(text, size, "internal error in '%s': sensor not found", name);
      return -32603;
    }
    magnetic_read_mgauss(mg);
    total = sqrtf((float_t)mg[0] * mg[0] + (float_t)mg[1] * mg[1] + (float_t)mg[2] * mg[2]);
    snprintf(text, size, "X %d mG, Y %d mG, Z %d mG, total %d mG", mg[0], mg[1], mg[2], (INT)(total + 0.5f));
    return 0;
  }
  snprintf(text, size, "unknown tool: %s", name);
  return -32601;
}

/* ---- Transport (TCP port 5555) ---- */

/* Send one JSON line to a peer on a new TCP connection (P2PTransport.send in Python). */
static void send_to_peer(const CHAR *peer_id, const CHAR *line)
{
  ULONG ip;
  UINT port;
  NX_PACKET *packet;
  UINT status;

  if (!peer_lookup(peer_id, &ip, &port))
  {
    log_printf("cannot reply to %s: unknown peer (no HELLO heard yet)", peer_id);
    return;
  }
  if (nx_tcp_client_socket_bind(&ClientSocket, NX_ANY_PORT, TX_TIMER_TICKS_PER_SECOND) != NX_SUCCESS)
  {
    log_printf("cannot reply to %s: socket busy", peer_id);
    return;
  }
  status = nx_tcp_client_socket_connect(&ClientSocket, ip, port, REPLY_TIMEOUT);
  if (status == NX_SUCCESS)
  {
    status = nx_packet_allocate(&AppPool, &packet, NX_TCP_PACKET, TX_TIMER_TICKS_PER_SECOND);
  }
  if (status == NX_SUCCESS)
  {
    status = nx_packet_data_append(packet, (VOID *)line, strlen(line), &AppPool, TX_TIMER_TICKS_PER_SECOND);
    if (status == NX_SUCCESS)
    {
      status = nx_tcp_socket_send(&ClientSocket, packet, REPLY_TIMEOUT);
    }
    if (status != NX_SUCCESS)
    {
      nx_packet_release(packet);
    }
  }
  if (status != NX_SUCCESS)
  {
    log_printf("reply to %s failed (NetX error 0x%02X)", peer_id, status);
  }
  nx_tcp_socket_disconnect(&ClientSocket, TX_TIMER_TICKS_PER_SECOND);
  nx_tcp_client_socket_unbind(&ClientSocket);
}

/* Handle one incoming message (the on_message chain in Python). */
static void handle_request(const CHAR *json)
{
  static CHAR reply[1536];                /* tools/list answer is about 900 bytes */
  CHAR from[PEER_ID_SIZE] = "?";
  CHAR type[32];
  CHAR method[32];
  CHAR name[40] = "";
  CHAR id[40];
  CHAR text[120];
  INT code;

  json_get_string(json, "from", from, sizeof(from));
  if (json_get_string(json, "type", type, sizeof(type)))
  {
    log_printf("ignored %s from %s (not used by a sensing-only device)", type, from);
    return;
  }
  if (!json_get_string(json, "method", method, sizeof(method)))
  {
    return;                               /* not JSON-RPC (e.g. console text broadcast) */
  }
  if (!json_get_raw(json, "id", id, sizeof(id)))
  {
    log_printf("%s from %s without id: notification, no reply", method, from);
    return;
  }

  if (strcmp(method, "tools/list") == 0)
  {
    snprintf(reply, sizeof(reply),
             "{\"jsonrpc\": \"2.0\", \"id\": %s, \"result\": {\"tools\": %s}, \"from\": \"%s\"}\n",
             id, TOOLS_JSON, AGENT_ID);
    log_printf("tools/list from %s -> 2 tools", from);
    send_to_peer(from, reply);
    return;
  }

  if (strcmp(method, "tools/call") == 0)
  {
    json_get_string(json, "name", name, sizeof(name));
    code = run_tool(name, text, sizeof(text));
  }
  else
  {
    snprintf(text, sizeof(text), "method not found: %s", method);
    code = -32601;
  }

  if (code == 0)
  {
    snprintf(reply, sizeof(reply),
             "{\"jsonrpc\": \"2.0\", \"id\": %s, \"result\": {\"content\": [{\"type\": \"text\", \"text\": \"%s\"}]}, \"from\": \"%s\"}\n",
             id, text, AGENT_ID);
  }
  else
  {
    snprintf(reply, sizeof(reply),
             "{\"jsonrpc\": \"2.0\", \"id\": %s, \"error\": {\"code\": %d, \"message\": \"%s\"}, \"from\": \"%s\"}\n",
             id, code, text, AGENT_ID);
  }
  log_printf("%s %s from %s -> %s", method, name, from, text);
  send_to_peer(from, reply);
}

/* Read one message: until newline or the sender closes. Returns its length (0 = nothing usable). */
static UINT tcp_receive_line(NX_TCP_SOCKET *socket, CHAR *buffer, UINT size)
{
  NX_PACKET *packet;
  UINT total = 0;

  buffer[0] = '\0';
  while (strchr(buffer, '\n') == NULL
         && nx_tcp_socket_receive(socket, &packet, 2 * TX_TIMER_TICKS_PER_SECOND) == NX_SUCCESS)
  {
    ULONG len = 0;

    if (total + packet->nx_packet_length >= size)
    {
      nx_packet_release(packet);
      log_printf("incoming message too long, dropped");
      return 0;
    }
    nx_packet_data_retrieve(packet, buffer + total, &len);
    nx_packet_release(packet);
    total += len;
    buffer[total] = '\0';
  }
  return total;
}

/* TCP server: one message per connection (P2PTransport._handle in Python). */
static VOID Agent_Thread_Entry(ULONG thread_input)
{
  static CHAR request[1024];

  nx_tcp_socket_create(&IpInstance, &ServerSocket, "Agent Server", NX_IP_NORMAL, NX_FRAGMENT_OKAY,
                       NX_IP_TIME_TO_LIVE, 2048, NX_NULL, NX_NULL);
  nx_tcp_socket_create(&IpInstance, &ClientSocket, "Agent Client", NX_IP_NORMAL, NX_FRAGMENT_OKAY,
                       NX_IP_TIME_TO_LIVE, 2048, NX_NULL, NX_NULL);
  if (nx_tcp_server_socket_listen(&IpInstance, AGENT_TCP_PORT, &ServerSocket, 5, NX_NULL) != NX_SUCCESS)
  {
    log_printf("TCP listen on port %u failed", AGENT_TCP_PORT);
    return;
  }

  while (1)
  {
    UINT len = 0;

    if (nx_tcp_server_socket_accept(&ServerSocket, TX_WAIT_FOREVER) == NX_SUCCESS)
    {
      len = tcp_receive_line(&ServerSocket, request, sizeof(request));
      nx_tcp_socket_disconnect(&ServerSocket, TX_TIMER_TICKS_PER_SECOND);
    }
    nx_tcp_server_socket_unaccept(&ServerSocket);
    nx_tcp_server_socket_relisten(&IpInstance, AGENT_TCP_PORT, &ServerSocket);
    if (len > 0)
    {
      handle_request(request);            /* after relisten: the next sender can already connect */
    }
  }
}

/**
* @brief  ip address change callback
* @param  ip_instance : NX_IP instance registered for this callback
* @param   ptr : VOID * optional data pointer
* @retval None
*/
static VOID ip_address_change_notify_callback(NX_IP *ip_instance, VOID *ptr)
{
  /* as soon as the IP address is ready, the semaphore is released to let the main thread continue */
  tx_semaphore_put(&Semaphore);
}


static VOID App_Main_Thread_Entry(ULONG thread_input)
{
  UINT ret;
  
  ret = nx_ip_address_change_notify(&IpInstance, ip_address_change_notify_callback, NULL);
  if (ret != NX_SUCCESS)
  {
    Error_Handler();
  }
  
  ret = nx_dhcp_start(&DHCPClient);
  if (ret != NX_SUCCESS)
  {
    Error_Handler();
  }
  
  /* wait until an IP address is ready */
  if(tx_semaphore_get(&Semaphore, TX_WAIT_FOREVER) != TX_SUCCESS)
  {
    Error_Handler();
  }
  /* get IP address */
  ret = nx_ip_address_get(&IpInstance, &IpAddress, &NetMask);
  
  PRINT_IP_ADDRESS(IpAddress);
  
  if (ret != TX_SUCCESS)
  {
    Error_Handler();
  }
  /* debug log lines to the PC (UDP port 9998, read with log_receiver.py) */
  nx_udp_socket_create(&IpInstance, &LogSocket, "Log Socket", NX_IP_NORMAL, NX_FRAGMENT_OKAY, NX_IP_TIME_TO_LIVE, 5);
  nx_udp_socket_bind(&LogSocket, NX_ANY_PORT, TX_WAIT_FOREVER);

  /* sensors: set up once, then only the agent thread reads them (one user of the I2C bus) */
  static CHAR msg[200];                   /* static: keeps it off the 2 KB thread stack */
  UINT bus_ok = sensors_bus_init();
  UINT hum_ok = 0;
  if (bus_ok)
  {
    i2c_scan(msg, sizeof(msg));
    udp_log(msg);
    TempOk = temperature_init();
    hum_ok = humidity_init();
    MagOk  = magnetic_init();
  }
  /* the sensors produce their first value about 1 s after setup */
  tx_thread_sleep(TX_TIMER_TICKS_PER_SECOND * 3 / 2);

  CHAR temp_text[16] = "NOT found";
  CHAR hum_text[16]  = "NOT found";
  INT mg[3] = {0};
  if (TempOk)
  {
    format_tenths(temp_text, sizeof(temp_text), temperature_read_tenths());
  }
  if (hum_ok)
  {
    format_tenths(hum_text, sizeof(hum_text), humidity_read_tenths());
  }
  if (MagOk)
  {
    magnetic_read_mgauss(mg);
  }
  log_printf("Sensors: I2C2 %s, temperature %s, humidity %s, magnetic %s (%d %d %d mG)",
             bus_ok ? "OK" : "FAIL", temp_text, hum_text, MagOk ? "OK" : "NOT found", mg[0], mg[1], mg[2]);

  /* sensing agent: discovery here, TCP requests in the agent thread */
  CHAR ip_text[16];
  ip_to_str(IpAddress, ip_text, sizeof(ip_text));
  tx_mutex_create(&PeersMutex, "Peers", TX_NO_INHERIT);
  nx_udp_socket_create(&IpInstance, &DiscoverySocket, "Discovery", NX_IP_NORMAL, NX_FRAGMENT_OKAY, NX_IP_TIME_TO_LIVE, 8);
  if (nx_udp_socket_bind(&DiscoverySocket, DISCOVERY_PORT, TX_WAIT_FOREVER) != NX_SUCCESS)
  {
    log_printf("UDP bind on port %u failed", DISCOVERY_PORT);
  }
  tx_thread_create(&AgentThread, "Agent thread", Agent_Thread_Entry, 0, AgentThreadStack, sizeof(AgentThreadStack),
                   DEFAULT_MAIN_PRIORITY, DEFAULT_MAIN_PRIORITY, TX_NO_TIME_SLICE, TX_AUTO_START);
  log_printf("Agent %s at %s: TCP %u, discovery UDP %u, no leader preset", AGENT_ID, ip_text, AGENT_TCP_PORT, DISCOVERY_PORT);

  ULONG last_hello = tx_time_get() - ANNOUNCE_INTERVAL;   /* announce right away */
  ULONG last_status = tx_time_get();
  while (1)
  {
    if (tx_time_get() - last_hello >= ANNOUNCE_INTERVAL)
    {
      send_hello();
      last_hello = tx_time_get();
    }
    discovery_receive(TX_TIMER_TICKS_PER_SECOND / 10);   /* also paces this loop */
    peers_expire();
    if (tx_time_get() - last_status >= 30 * TX_TIMER_TICKS_PER_SECOND)
    {
      log_printf("alive, %u peer(s)", peer_count());
      last_status = tx_time_get();
    }
  }
}
/* USER CODE END 1 */
