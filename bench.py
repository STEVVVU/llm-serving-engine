import argparse
import json
from pathlib import Path
import random
import statistics
from time import perf_counter
import torch
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Bounded synthetic offline serving throughput, with actual output counts.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--input-length", type=int, default=24)
    parser.add_argument("--output-length", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("evidence/throughput.json"))
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.input_length, args.output_length, args.repeats) <= 0:
        parser.error("workload sizes must be positive")
    torch.set_num_threads(2)
    context = args.input_length + args.output_length
    engine = LLM(args.model, device=args.device, enforce_eager=args.eager, compile=args.compile,
        max_num_seqs=args.concurrency, max_num_batched_tokens=max(args.input_length, args.concurrency),
        max_model_len=context, kvcache_block_size=4,
        num_kvcache_blocks=args.concurrency * ((context + 3) // 4))
    rng = random.Random(17)
    vocab = engine.config.hf_config.vocab_size
    prompts = [[rng.randrange(3, vocab) for _ in range(args.input_length)] for _ in range(args.requests)]
    params = SamplingParams(temperature=0, max_tokens=args.output_length, ignore_eos=True)
    runs = []
    try:
        engine.generate(prompts[:args.concurrency], params, False)
        for _ in range(args.repeats):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            outputs = engine.generate(prompts, params, False)
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds = perf_counter() - start
            count = sum(len(output["token_ids"]) for output in outputs)
            assert len(outputs) == args.requests and count == args.requests * args.output_length
            runs.append(dict(seconds=seconds, output_tokens=count, requests=len(outputs), errors=0,
                             tokens_per_second=count / seconds))
        evidence = dict(workload=vars(args) | {"output": str(args.output)}, torch=torch.__version__,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
            cache_policy="persistent radix cache, unmeasured initial concurrency-sized warmup",
            measurement="end-to-end offline generate, tokenization excluded for token-ID prompts, sampling and scheduler included",
            runs=runs, median_tokens_per_second=statistics.median(r["tokens_per_second"] for r in runs),
            range_tokens_per_second=[min(r["tokens_per_second"] for r in runs), max(r["tokens_per_second"] for r in runs)],
            pages=engine.scheduler.block_manager.stats(),
            graph_replays=getattr(engine.model_runner, "graph_replays", 0),
            claim="synthetic local sample only; no baseline comparison or course performance claim")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, default=str))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
