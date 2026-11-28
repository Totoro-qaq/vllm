# Inkling 2 Mini prefill on desktop RTX 5090

Two independent changes share the same baseline and workload. The source commits belong to vllm-project/vllm and its contributor fork.

| Arm | Python source commit |
| --- | --- |
| Stock | 9eb918cab166895a5a00b9d847f1180fceb8217f |
| Native MXFP8 experts | 7fa33f848022c582221bd226d59d9579d07c3fa8 |
| Fused loop controller | 7d1092cc3ec56be97d376c639ae0d189c008e83f |

The run used one RTX 5090 32 GB, driver 595.71.05, PyTorch 2.14.0+cu132 and Triton 3.8.0. All three source trees used the same compiled vLLM extensions from the CUDA 13.2 nightly at 54b713f990214bce1fda519caa89e15523828776. The benchmarked Python modules were loaded from each source tree. The checkpoint was thinkingmachines/Inkling-2-Mini at c9d028302cd0f035fdf8e7e91f089a4ce94d2195.

## Reproduction

Run from an environment with the corresponding vLLM revision and a local copy of the pinned model. The client starts and stops its own streaming server on loopback port 18591. It sets TP1, a 16,384-token context limit, 4,096 max batched tokens, at most eight sequences, FP8 KV, 94% GPU memory utilization, text-only serving, the Inkling reasoning parser and disabled prefix caching. The measured requests are serial, with a greedy target and one streamed output token.

```bash
uv run --no-project --python "$VIRTUAL_ENV/bin/python" "$VIRTUAL_ENV/bin/python" bench.py \
  --model /path/to/Inkling-2-Mini --out run-1 --quality
```

The six fresh-process runs were stock, MXFP8, controller, controller, MXFP8, stock. Each ran three warm-ups at each length and ten measured requests per length in seeded, shuffled order. The figures show the average of the two process medians per arm. Both implementations use the original path for 512 tokens. Run with `--quality` for the first process per arm to collect the 48 seeded engineering cases (long SQL, code tracing and document extraction) and six fixed-prefix logprob probes. These are small behavioral checks on one model and device.

[Per-request timing](data/ttft.csv) · [timing and quality summary](data/summary.json) · [raw sampled answers and fixed-prefix probabilities](data/) · [figures](figures/)

To regenerate the two figures from the published measurements, run the plotting script with matplotlib installed.

```bash
uv run --no-project --with matplotlib --python .venv/bin/python draw_figures.py data/summary.json
```
