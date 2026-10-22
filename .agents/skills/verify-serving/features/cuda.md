# CUDA decode replay

## Sub-features

Covering graph buckets, padded scratch lanes, dynamic context lengths and page tables, greedy parity and pressure reuse.

## How to get to it (user POV)

Use CUDA-enabled Torch and `LLM(..., device="cuda")`. Use `enforce_eager=True` for the comparison path.

## Driving it with Python

Run `python tools/verify_serving.py --device cuda --model PATH --output evidence/cuda-runtime.json`. Acceptance requires eager equality, independent HF equality, a recorded `(3, 4)` live-lane and graph-bucket pair, actual replay counts, cache eviction, HTTP cancellation and zero active pages after completion.

## Gotchas

This path uses portable Torch operations in captured CUDA graphs. Native flash-attn and NCCL and optional compilation require separate environment checks. The result makes no speedup claim.
