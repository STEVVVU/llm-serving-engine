# LLM serving engine

## Commands

Run from this directory with Python 3.10 through 3.12 and the dependencies in
`pyproject.toml`. `requirements-verified.txt` pins the tested package versions.

- Install with `python -m pip install -e . -c requirements-verified.txt`.
- Run regressions with `python -m unittest discover -s tests -v`.
- Run the real CPU product driver with `python tools/verify_serving.py --device cpu --output evidence/cpu-runtime.json`.
- Run real-model CUDA proof with `python tools/verify_serving.py --device cuda --model PATH --output evidence/cuda-runtime.json`.
- Check syntax with `python -m compileall -q nanovllm tools tests bench.py`.
- Check whitespace with `git diff --check` and include newly staged files.
- Start the local service with `python -m nanovllm.server --model PATH`.
- Measure the bounded synthetic workload with `python bench.py --model PATH --device cuda --concurrency 3 --requests 8 --input-length 24 --output-length 8 --repeats 5`.

## Architecture and conventions

`nanovllm.engine.llm_engine.LLMEngine` preserves the public offline entry points.
The default portable `TorchRunner` executes the upstream Qwen3 model and paged
attention with Torch operations. The optional `ModelRunner` uses Linux CUDA,
flash-attn, Triton and NCCL. Native imports stay lazy so CPU users can import the
package. The native backend and optional compilation remain outside local proof.

Sequences own page references. Exact full-page radix nodes own cached KV pages.
The scheduler bounds token work, live admission and maximum request footprints.
`server.Service` is the sole owner of its engine. HTTP handlers exchange bounded
messages and cancellation signals with that owner. Attention context is still
process-global, so engine operations must remain on one owner thread at a time.

## Verification and sources

Use `.agents/skills/verify-serving/SKILL.md` and its feature map for launch,
doctor, drive, retained evidence and cleanup. The maintained driver creates tiny
local safetensors for CPU proof, launches only its own child server and preserves
JSON evidence under ignored `evidence/`. Real model checks require an existing
local checkpoint. Regression assertions compare deterministic logits, tokens and
lifecycle invariants. Generated prose is not a golden value.

`docs/provenance.md` identifies the MIT upstream source, primary coursework
references and verified limits. Retain its attribution and the LICENSE. Runtime
correctness, synthetic throughput, hosted CI and standardized course grading are
distinct claims. Update affected feature-map entries when a user path changes.
