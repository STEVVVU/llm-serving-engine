# HTTP request lifecycle

## Sub-features

Concurrent completion, custom SSE events, explicit cancellation, body limits, bounded mailboxes, admission and terminal cleanup.

## How to get to it (user POV)

Run `python -m nanovllm.server --model PATH` on numeric IPv4 loopback. POST JSON to `/v1/completions`. GET `/metrics` to inspect conserved pages and request totals.

## Driving it with Python

The maintained `tools/verify_serving.py` launches its own child server, waits for `/health`, sends concurrent requests, consumes an SSE completion, cancels a live request through DELETE, checks rejected inputs and proves accepted equals completed plus cancelled plus failed. The unit suite drives actual model-owner backpressure and a controlled terminal queue consumer race.

## Gotchas

SSE terminal events contain the full result. Intermediate events contain token deltas. This is not the complete OpenAI streaming schema. Slow writers time out after five seconds. The worker cancels clients whose mailbox fills without blocking other inference.
