# LLM serving engine

An MIT-licensed extension of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), preserving its full Git history, license, Qwen3 model and offline API. This project adds a portable PyTorch paged-attention backend, bounded scheduling, exact page radix caching, and a local HTTP server. [Provenance and limitations](docs/provenance.md) describe the coursework relationship and source boundaries. The [original README](docs/upstream-README.md) retains upstream descriptions and benchmark attribution.

## Run locally

Use Python 3.10 through 3.12 and a local Qwen3 safetensors checkpoint. The loader never downloads weights or loads pickle checkpoints.

```bash
python -m pip install -e .
python -m nanovllm.server --model /path/to/qwen3 --device cpu --eager
```

For a CUDA-enabled PyTorch installation, use `--device cuda`. Decode graphs are enabled on CUDA unless `--eager` is set. The server defaults to four concurrent running sequences, a 64-token step budget, 64 pages of 16 tokens, 16 HTTP clients, 16 queued client events, a 64 KiB body limit, and a 120-second request deadline. Requests must fit both the model context and the fixed page pool. The server binds to loopback by default.

```bash
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt":"Hello","max_tokens":16,"temperature":0,"stream":true}'
```

`POST /v1/chat/completions` accepts text messages through the model's local chat template. This is a small JSON API, with custom SSE token events and a final full-result event followed by `[DONE]`. It does not claim full OpenAI protocol compatibility. Cancel a streaming request with `DELETE /requests/<request_id>` using the ID from its first event. `/health` and `/metrics` expose bounded lifecycle and page statistics.

The CLI accepts only numeric IPv4 loopback addresses for `--host`.

## Offline API

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/qwen3", device="cuda", max_num_seqs=4,
          max_model_len=512, kvcache_block_size=16, num_kvcache_blocks=64)
try:
    outputs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=16))
    print(outputs[0]["text"])
finally:
    llm.exit()
```

`add_request` returns a sequence ID. `step` preserves the upstream `(finished_outputs, signed_token_count)` result. `cancel_request` releases that request's page references. `generate` refills the bounded queue for larger input lists. Temperature zero selects greedy decoding, and zero requested tokens complete immediately. Each engine requires one owner thread because upstream attention context is process-global.

## Mechanisms

Packed prefill runs projections and MLPs on real tokens only. Attention scatters KV into physical pages and gathers through page tables. Absolute query positions preserve causal masking across cached prefixes and prefill chunks. The token budget bounds both prefill and decode work. Eligible prefill and decode phases alternate, and budget-limited decode rotates live requests. Admission conservatively reserves each live request's maximum KV footprint so all admitted requests can finish without preemption.

The radix tree stores exact full-page token tuples under their exact parent. Complete processed generation pages enter the same tree as prompt pages. Sequence references pin pages. Eviction removes only unpinned LRU leaves. Cached pages remain reusable after completion, while active page references return to zero. Decode graphs use stable buffers and covering batch buckets. Padding writes to reserved scratch slots, and every replay replaces all page-table columns and inactive lane metadata.

The original native attention kernels remain in `native_attention.py`. Install `.[native]` on a compatible Linux CUDA system and select `backend="native", device="cuda"` to use the upstream flash-attn, Triton and NCCL path. That backend and tensor parallelism have not been executed in this Windows validation. `compile=True` or `--compile` opts into PyTorch compilation of norm, rotary, activation and sampling regions. Compiler availability and compile performance remain unverified here.

## Verify and measure

The maintained [verification skill](.agents/skills/verify-serving/SKILL.md) records launch, doctor, drive, evidence and cleanup. It covers independent Hugging Face forward parity, page conservation, cache pressure, fairness, slow-client cancellation and real HTTP lifecycles.

```bash
python -m unittest discover -s tests -v
python tools/verify_serving.py --device cpu --output evidence/cpu-runtime.json
python tools/verify_serving.py --device cuda --model /path/to/qwen3 --output evidence/cuda-runtime.json
python bench.py --model /path/to/qwen3 --device cuda --concurrency 3 --requests 8 --input-length 24 --output-length 8 --repeats 5
```

The benchmark records actual output counts, errors, per-run times, hardware and cache policy. It includes scheduler, sampling and model execution in `generate`, with an unmeasured warmup. Synthetic token-ID prompts exclude tokenization. Compare configurations with interleaved repeated runs and representative prompts before making a performance claim. Local sample evidence is not a standardized benchmark, a speedup result, or completion of Stanford's grader.

The retained [CUDA verification](docs/results/cuda-verification.json) records independent HF greedy parity, eager/graph parity, three live requests in a four-lane graph bucket, cache eviction and HTTP cancellation with zero active pages afterward. The [five-run local measurement](docs/results/cuda-throughput.json) retains all timings, output counts and errors. These copies normalize machine-specific model/executable paths; local originals remain unchanged. The later telemetry-storage correction changes no numerical execution and was checked with the CPU replay regression and HTTP driver. Hosted CI runs CPU checks and uploads its actual HTTP evidence; it does not run the pretrained CUDA workload.
