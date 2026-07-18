# TTFT measurements

> Back to [Latency and network README](../README.md)

Time-to-first-token benchmarks for the two leader inference backends. TTFT is the
time from starting generation to the first streamed token (prefill included). The
first inference of a run reads higher (cold model load).

These scripts import the live leader code (`LeaderLogic`, models under
`models/`), so run each on its target board **from the `IMPLEMENTATION/` folder**,
not from here.

| Script | Board | What it does |
|---|---|---|
| [`ttft_qwen3_hailo.py`](ttft_qwen3_hailo.py) | Raspberry Pi 5 + Hailo | Sends 5 person queries through the live Hailo dispatch input (Workflow1: system prompt + tools, native `.hef` streaming) and logs each TTFT to `ttft_qwen3_hailo.json`. |
| [`ttft_functiongemma_stm32.py`](ttft_functiongemma_stm32.py) | STM32MP257F-DK | Sweeps the dispatch tool set (1, 6, 12, 18, 24) and runs 10 person queries through the two-stage leader pipeline (dispatch -> mock tool -> answer), calling `llm.reset()` before every generation (cache-cold prefill). Writes `ttft_functiongemma_stm32.json`. |
| [`ttft_functiongemma_stm32_USE_KV_CACHE.py`](ttft_functiongemma_stm32_USE_KV_CACHE.py) | STM32MP257F-DK | Same sweep and metrics, but **without** `llm.reset()` between generations, so llama.cpp reuses the longest common prefix already in the KV cache (the deployed leader's real behavior). TTFT becomes order-dependent. Writes `supertest_stm32_functiongemma.json`. |

Per query the functiongemma scripts record ttft, dispatch/answer token counts,
decode tk/s, per-stage latencies and end-to-end time; the Hailo script records
ttft per message.
