---
name: verify-serving
description: Drive nano-vllm offline and HTTP serving with local safetensors, CPU parity and optional CUDA graph evidence.
---

## Launch

Use Python 3.11 with dependencies in pyproject.toml. From the repository root run
`python tools/verify_serving.py --output evidence/runtime.json --device cpu`.
The driver owns its temporary tiny Qwen checkpoint and server subprocess.
For the cached real model use `--model <local-safetensors-directory> --device cuda`.
No downloads, pickle loading, authentication, or external endpoints are involved.

## Doctor

Run `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
The driver waits for its own /health endpoint and fails if the process exits.
Onboarding was proved on CPU and CUDA with offline parity and a child HTTP
server. The upstream baseline import failed on missing xxhash before changes.
The portable backend no longer imports xxhash, Triton, or flash-attn.

## Drive

For framework adapters and resource grants, follow [features/metrics.md](features/metrics.md). Native HF-SDPA/paged eager/graph runs passed exact token, backend, EOS and page checks at two offline workload points: 128/32/concurrency-4 and 256/64/concurrency-1. Preserve earlier API/disk failures and the untuned HF-eager control. These two points do not establish service capacity or a causal performance winner.

Run `python -m unittest discover -s tests -v` for independent HF-forward parity,
page lifecycle and scheduling regressions. Run the driver above for real offline
generation, concurrent HTTP completion, SSE, cancellation and page conservation.
See features/README.md for the supported paths.

## Evidence

The driver records requests, responses, engine statistics and cleanup in the
requested JSON file. Keep generated evidence outside maintained tooling. Raw
token IDs prove deterministic parity; generated prose is never a golden value.

## Cleanup

The driver terminates only its child server in a finally block and removes its
temporary checkpoint. Evidence survives cleanup. Check the output file exists.
