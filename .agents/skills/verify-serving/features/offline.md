# Offline generation

## Sub-features

`generate`, queue refill, incremental `add_request` and `step`, zero-token requests, cancellation and capacity rejection.

## How to get to it (user POV)

Import `LLM` and `SamplingParams` from `nanovllm` with a local Qwen3 directory.

## Driving it with Python

Run `python tools/verify_serving.py --device cpu --output evidence/cpu-runtime.json`.
Run `python -m unittest discover -s tests -v` for bounded queue refill and page lifecycle regressions.

## Gotchas

Every request reserves its maximum output footprint. An impossible request raises `ValueError` before admission. Use one owner thread per engine.
