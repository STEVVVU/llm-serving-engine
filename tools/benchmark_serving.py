"""Torch Timer comparison of native HF generate and the existing paged engine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SIDES = ("hf", "eager", "graphs")


def trial_order(repeat):
    offset = repeat % len(SIDES)
    order = SIDES[offset:] + SIDES[:offset]
    return order[::-1] if repeat // len(SIDES) % 2 else order


def generation_options(count, pad_token_id):
    # Explicit None kwargs take precedence over pretrained EOS defaults in Transformers 5.20.2.
    return dict(do_sample=False, num_beams=1, max_new_tokens=count, min_new_tokens=count,
                eos_token_id=None, forced_eos_token_id=None, pad_token_id=pad_token_id,
                use_cache=True, cache_implementation="dynamic", disable_compile=True,
                return_dict_in_generate=True, output_scores=False)


def assert_tokens(actual, expected, count):
    if len(actual) != len(expected) or any(len(row) != count for row in actual) or actual != expected:
        raise AssertionError("generation length/token mismatch against independent full-forward oracle")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.input_length <= 256 and 1 <= args.output_length <= 64 and 1 <= args.concurrency <= 16):
        parser.error("bounded offline matrix requires input 1..256, output 1..64 and concurrency 1..16")
    model_path = Path(args.model).resolve()
    if model_path.name != MODEL_REVISION:
        parser.error(f"require cached Qwen3 snapshot {MODEL_REVISION}")
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 10 * 1024 ** 3:
        raise RuntimeError("preserve 10 GiB free disk")
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    from torch.utils.benchmark import Timer
    from transformers import AutoTokenizer, GenerationConfig, Qwen3ForCausalLM, StoppingCriteriaList
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.scheduler import Scheduler
    from test_runtime import reference_tokens

    torch.set_num_threads(2)
    n, length, count, concurrency = (2, 16, 2, 2) if args.preflight else (16, args.input_length, args.output_length, args.concurrency)
    rows = [[100 + i] + [300 + (j % 50) for j in range(length - 1)] for i in range(n)]
    report = {"passed": False, "preflight_only": args.preflight, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                  [Path(__file__), ROOT / "tests/test_runtime.py", *sorted((ROOT / "nanovllm").rglob("*.py"))]},
              "model_file_sha256": {p.name: file_hash(p) for p in sorted(model_path.iterdir()) if
                  p.suffix == ".safetensors" or p.name in ("config.json", "tokenizer.json", "tokenizer_config.json")},
              "command": sys.argv, "model": args.model, "model_revision": MODEL_REVISION,
              "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
              "numerics": "BF16, identical local Qwen weights; exact greedy parity with the independent eager oracle is required",
              "attention_backends": {"oracle": "Transformers eager, single-request full-forward greedy",
                  "hf": "Transformers sdpa with automatic CUDA dispatch and native DynamicCache",
                  "eager": "paged Torch scaled_dot_product_attention with automatic CUDA dispatch",
                  "graphs": "same paged Torch SDPA, CUDA graphs for decode",
                  "profile": "separate paged graphs arm; not native HF or paged eager"},
              "workload": {"requests": n, "input_tokens": length, "output_tokens_per_request": count,
                           "max_live_batch": concurrency, "paged_prefill_token_budget": concurrency * length},
              "cache": "warm weights/graphs; no cross-request prefix reuse; fresh HF DynamicCache per batch",
              "timing_scope": "generate, text decode, exact token check; paged scheduler reset included per invocation",
              "framework": "torch.utils.benchmark.Timer.timeit(number=1), two automatic warmups per sample",
              "order": [], "runs": [], "verified_calls": {side: 0 for side in SIDES}}
    engines, teacher = {}, None
    latest = {}

    def check_resources():
        if torch.cuda.max_memory_allocated() > 16 * 1024 ** 3 or shutil.disk_usage(args.output).free < 10 * 1024 ** 3:
            raise RuntimeError("bounded memory/disk limit exceeded")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        teacher = Qwen3ForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                    use_safetensors=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
        # Remove pretrained sampling defaults, then pass EOS=None explicitly at the highest precedence.
        teacher.generation_config = GenerationConfig()
        options = generation_options(count, tokenizer.pad_token_id)
        effective, unused = teacher._prepare_generation_config(None, **options)
        effective.max_length = length + count
        teacher._prepare_special_tokens(effective, kwargs_has_attention_mask=True, device="cuda", batch_size=concurrency)
        processors = teacher._get_logits_processor(effective, input_ids_seq_length=length, device="cuda", model_kwargs={})
        stopping = teacher._get_stopping_criteria(effective, stopping_criteria=StoppingCriteriaList())
        if unused or effective._eos_token_tensor is not None or processors or [type(x).__name__ for x in stopping] != ["MaxLengthCriteria"]:
            raise AssertionError("HF effective generation must preserve greedy logits and stop only at fixed length")
        report["hf_generation_audit"] = {"kwargs": options, "effective_eos_token_id": effective.eos_token_id,
            "logits_processors": [type(x).__name__ for x in processors], "stopping_criteria": [type(x).__name__ for x in stopping]}
        expected = [reference_tokens(teacher, row, count, "cuda") for row in rows]
        (args.output / "oracle.json").write_text(json.dumps({"prompts": rows, "expected": expected, "output_tokens": count}), encoding="utf-8")
        # Reuse this model after the independent eager oracle; do not load a fourth copy.
        teacher.set_attn_implementation("sdpa")
        if teacher.config._attn_implementation != "sdpa":
            raise AssertionError("native HF comparator did not switch to SDPA")
        params = SamplingParams(temperature=0, max_tokens=count, ignore_eos=True)
        for side in ("eager", "graphs"):
            context = length + count
            engines[side] = LLM(args.model, device="cuda", enforce_eager=side == "eager", max_model_len=context,
                               max_num_seqs=concurrency, max_num_batched_tokens=concurrency * length,
                               kvcache_block_size=16, num_kvcache_blocks=concurrency * ((context + 15) // 16))
            if next(engines[side].model_runner.model.parameters()).dtype != torch.bfloat16:
                raise AssertionError("all arms require identical BF16 numerics")
        check_resources()
        report["initialization_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()

        def hf_generate():
            actual, cache_bytes = [], 0
            with torch.inference_mode():
                for start in range(0, n, concurrency):
                    ids = torch.tensor(rows[start:start + concurrency], device="cuda")
                    output = teacher.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **options)
                    actual.extend(output.sequences[:, length:].tolist())
                    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                                for layer in output.past_key_values.layers for tensor in (layer.keys, layer.values)
                                if tensor is not None}
                    cache_bytes = max(cache_bytes, sum(storages.values()))
                    del output
            tokenizer.batch_decode(actual)
            assert_tokens(actual, expected, count)
            latest["hf"] = {"dynamic_cache_peak_batch_bytes": cache_bytes, "max_live_batch": min(n, concurrency)}
            report["verified_calls"]["hf"] += 1

        def paged_generate(side):
            engine = engines[side]
            if not engine.is_finished():
                raise AssertionError("previous workload still active")
            # Reset inside every invocation, including both Timer warmups, to forbid reuse.
            engine.scheduler = Scheduler(engine.config)
            actual = engine.generate(rows, params, False)
            assert_tokens([row["token_ids"] for row in actual], expected, count)
            pages = engine.scheduler.block_manager.stats()
            if pages["prefix_hits"] or pages["active_pages"] or pages["total_pages"] != sum(pages[k] for k in ("free_pages", "active_pages", "cached_pages")):
                raise AssertionError("prefix policy or page conservation violated")
            latest[side] = {"kv_tensor_bytes": engine.model_runner.kv_cache.untyped_storage().nbytes(), "pages": pages}
            report["verified_calls"][side] += 1

        runners = {"hf": hf_generate, "eager": lambda: paged_generate("eager"), "graphs": lambda: paged_generate("graphs")}
        for repeat in range(1 if args.preflight else 5):
            order = trial_order(repeat)
            report["order"].append(order)
            for side in order:
                check_resources()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()
                measured = Timer(stmt="generate()", globals={"generate": runners[side]}, num_threads=2).timeit(number=1)
                peak = torch.cuda.max_memory_allocated()
                check_resources()
                report["runs"].append({"repeat": repeat, "side": side, "seconds": measured.median,
                    "output_tokens": n * count, "tokens_per_second": n * count / measured.median,
                    "offered": n, "completed": n, "errors": 0, "all_models_resident_cuda_bytes": resident,
                    "all_models_peak_cuda_bytes": peak, "incremental_cuda_peak_bytes": peak - resident, **latest[side]})
        if args.profile:
            engine = engines["graphs"]
            original = engine.model_runner.run
            def profiled(sequences, prefill):
                with torch.profiler.record_function("prefill" if prefill else "decode"):
                    return original(sequences, prefill)
            engine.model_runner.run = profiled
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], profile_memory=True) as profile:
                    paged_generate("graphs")
                profile.export_chrome_trace(str(args.output / "profile.json"))
                (args.output / "profile.txt").write_text(profile.key_averages().table(sort_by="self_device_time_total", row_limit=30), encoding="utf-8")
            finally:
                engine.model_runner.run = original
        report["peak_process_cuda_bytes"] = max(report["initialization_peak_cuda_bytes"],
            torch.cuda.max_memory_allocated(), *(row["all_models_peak_cuda_bytes"] for row in report["runs"]))
        report["passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            report["peak_process_cuda_bytes"] = max(report.get("peak_process_cuda_bytes", 0),
                report.get("initialization_peak_cuda_bytes", 0), torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else None
        except (AssertionError, RuntimeError) as resource_error:
            report["peak_process_cuda_bytes"] = None
            report["cuda_resource_error"] = str(resource_error)
        for engine in engines.values():
            engine.exit()
        del teacher
        report["cleanup"] = "owned engines exited; allocations released on process exit"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
