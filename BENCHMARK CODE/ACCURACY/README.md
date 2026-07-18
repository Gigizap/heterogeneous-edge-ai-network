# LLM Tool-Call Benchmark

Benchmarks were run on a PC with an NVIDIA 3080 Laptop GPU (16GB VRAM), Intel i9-12900H, and 32GB RAM.

Greedy decoding is used throughout (temperature = 0.0) with a fixed random seed (7) and a repetition penalty of 1.1 to mitigate the degenerate looping that greedy decoding can induce in hybrid-reasoning models. Determinism is guaranteed within a fixed build and hardware configuration, and generated-token counts remain directly comparable across models.

The evaluation has three parts:

Part A measures single-tool dispatch over the full tool set, with pool sizes of 4, 8, 12, 16, 20, and 24 tools. For each query the ground-truth tool is guaranteed to be in the pool; remaining slots are filled by uniform random sampling under the fixed seed. Accuracy is reported for tool selection (correct name) and full tool call (correct name + all parameters), separately for queries expecting a tool and queries expecting abstention.

Part B restricts the universe to zero-parameter tools and compares the language models against the Jina reranker (jina-reranker-v2-base-multilingual) on selection accuracy alone, with pools of 4, 8, 12, and 16 tools.

Part C measures summarization from tool results: each model is shown 1, 5, 10, 15, and 20 function replies under two system prompts, and mean generated-token counts are recorded. Reply subsets are nested via a per-query seeded shuffle so smaller counts are strict subsets of larger ones.

Tool-call generation is capped at 256 tokens for dispatch and 512 tokens for summarization. Parameter matching is order-independent over keys and applies light normalization (numeric coercion and case-insensitive, whitespace-trimmed string comparison). All pool assignments and reply samples are cached to disk and reused across runs.

## requirements

```
pip install -r requirements.txt
```

`llama-cpp-python` needs CUDA support for GPU inference

You also need Ollama installed with `glm-4.7-flash` pulled.

## how to run

1. Put the models in `models/`: `Qwen3-1.7B-Q4_K_M.gguf` and `functiongemma-270m-it-Q4_K_M.gguf`. 
2. Then run `COMPLETE_BENCHMARK.py`
3. Set up Ollama: pull `glm-4.7-flash` and run `ollama serve`
4. Run `GRADE_PART_C.py`
5. If some questions are not properly graded, run `ANNOTATE_PENDING.py`
6. Run `MAKE_REPORT.py`
7. Run `GRADE_PART_C_PLOTTING.py`
8. Results are in `results/` and `results/grading/` and `results/report/`
