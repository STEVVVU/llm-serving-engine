# Serving feature map

| Feature | Driver | Proof |
|---|---|---|
| [Offline generate and step](offline.md) | tools/verify_serving.py | Generated token IDs and released pages |
| [Packed/chunked paged inference](paged-inference.md) | tests/test_runtime.py | Independent HF logits and greedy IDs |
| [Prefix sharing and eviction](paged-inference.md) | tests/test_runtime.py | Exact radix reuse, pins, conservation |
| [HTTP and SSE lifecycle](http.md) | tools/verify_serving.py | Concurrent results, terminal events, cancellation |
| [CUDA decode graphs](cuda.md) | tools/verify_serving.py --device cuda | Eager parity, three live lanes in bucket four |
| [Framework measurements](metrics.md) | Torch Timer/profiler and Locust adapters | Two offline HF-SDPA/paged eager/graphs workload points passed exact checks; SSE preflight passed |

## Sub-features

Each row above is a bounded acceptance path. HTTP metrics expose pages, queue
depth, graph replays and accepted request cleanup.

## How to get to it (user POV)

Import `LLM` and `SamplingParams`, or run `python -m nanovllm.server --model PATH`.

## Driving it with Python

Invoke the maintained driver and unittest commands in the parent skill.

## Gotchas

Native Linux flash-attn/NCCL requires separate optional dependencies. CUDA checks
require a CUDA PyTorch installation and enough device memory. No performance or
course-grader result follows from correctness evidence.
