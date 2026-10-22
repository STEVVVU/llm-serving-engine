# Paged inference and cache

## Sub-features

Packed projections, absolute-position chunk masks, exact page radix reuse, generated-page reuse, pinned references and LRU leaf eviction.

## How to get to it (user POV)

Use `kvcache_block_size`, `num_kvcache_blocks` and `max_num_batched_tokens` in `LLM` configuration.

## Driving it with Python

Run `python -m unittest discover -s tests -v`. The independent HF forward checks logits for every scheduled mixed-length chunk and compares greedy token IDs. The driver records pressure, cache hits, eviction and total-page conservation.

## Gotchas

Only complete processed pages enter the cache. The final generated token has no KV until another forward processes it. Cached unpinned pages count separately from free and active pages.
