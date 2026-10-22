# Offline Timer matrix status

Status: **2 of 24 cells measured** on 2026-10-22; 22 remain unrun. The retained points are 128 input / 32 output / concurrency 4 and the long case 256 / 64 / concurrency 1. See the [raw-verified summary](evidence/torch-metrics-20261022/sdpa-summary.json), which includes exact result, oracle and supervisor receipts for both. These are two offline synthetic workload observations; they establish no 7B behavior, TTFT, live service capacity, general framework ranking or causal winner. The full matrix still needs separate serialized resource grants.

Reuse `tools/benchmark_serving.py` unchanged at SHA-256 `eada4dd0ee14f0ec0ede9612a9663ac01d995fefd507ba40f3fbae0369dca2d5`. There is no new generation, timing or scheduling engine. The two completed cases use the existing private metrics environment, offline cached Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, BF16 weights, native HF SDPA, paged eager SDPA and paged decode graphs. Both retain independent single-request eager full-forward token oracles and 15 exact-verified calls per arm (five measured Timer samples and two automatic warmups per sample). Their raw EOS/logits and page checks passed; keep any future failure receipt intact and do not introduce tolerance or discard a failed cell.

## Frozen proposed workload

The Cartesian product is input lengths **32, 128, 256**, output counts **16, 64**, and maximum live batches **1, 2, 4, 8**: 24 fresh-process cases. Each case keeps 16 fixed token-ID prompts with unique first tokens. It uses the existing five rotated trials per arm, two automatic Timer warmups per trial, greedy generation without EOS stopping or suppression, fresh native DynamicCache per batch and no paged prefix reuse. The paged prefill budget is concurrency times input length; pool capacity follows the same supported maximum-context reservation. Model loading and oracle construction are excluded from samples. Input transfer, paged scheduler reset, returned-text decoding and exact token checks retain their current timing boundaries.

The 256 / 64 / 1 case has completed under its 1,800-second cap. The 128 / 32 / 4 case also completed. Run remaining cells in the table order below, skipping both recorded cells. Cell order does not change fixed within-case arm rotation. A cap hit remains a failed receipt, not a shortened workload or a timing sample.

| Input tokens | Output tokens | Concurrency values | External cap per case |
|---:|---:|---|---|
| 32 | 16 | 1, 2, 4, 8 | 900 s |
| 128 | 16 | 1, 2, 4, 8 | 900 s |
| 256 | 16 | 1, 2, 4, 8 | 900 s |
| 32 | 64 | 1, 2 / 4, 8 | 1,800 s / 900 s |
| 128 | 64 | 1, 2 / 4, 8 | 1,800 s / 900 s |
| 256 | 64 | 1, 2 / 4, 8 | 1,800 s / 900 s |

The total worst-case process budget is **18 × 900 + 6 × 1,800 = 27,000 seconds (7.5 hours)**, before review/cleanup. This is an upper bound, not a request to start an unattended campaign. The first case and subsequent serialized batches each need a grant. Retain all completed and failed cells rather than publish a selected subset.

## Invocation and resource boundary

After a grant, invoke one cell using the existing owned-tree supervisor and substitute the table's input/output/concurrency/cap values:

```text
.metrics-venv/Scripts/python.exe tools/benchmark_serving.py --model C:/Users/steve/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca --output evidence/<fresh-matrix-id>-i256-o64-c1 --input-length 256 --output-length 64 --concurrency 1
```

Do not pass `--profile` to measured cells. The existing separate graph profile supports only the completed anchor and its unchanged paged source/config. If a new cell exposes a cliff, review its resource/correctness receipts and obtain a separate profile grant at that exact cell before a cause-dependent change. It cannot be diagnosed from the anchor profile alone.

Keep the adapter's 16 GiB Torch-allocated CUDA ceiling and a continuously monitored 10 GiB free-disk reserve. All three models remain resident; the largest proposed KV pool is larger than the anchor, so measured peaks must establish feasibility. The supervisor must cap and terminate only the owned process tree, record UTC start/end, source/model hashes, sampled tree RSS and disk minimum, and verify no owned GPU process remains after every case. Reuse cached weights without copying them. Receipts without traces are expected to be small, but check disk reserve rather than rely on that expectation.

## Reporting boundary

For each of the 24 cases retain all 15 arm samples, median/range, actual output tokens, errors, exact-call counts, cache/page receipts and memory. At 16 requests per arm per trial, each passing case contains 240 measured requests plus framework warmups. Do not pool unlike workloads into one throughput score. Offline batch completion does not measure HTTP admission, TTFT/ITL, offered-rate capacity, sustained service stability or language quality. Those remain separate studies using the maintained Locust adapter and an appropriate standard offered-rate framework.
