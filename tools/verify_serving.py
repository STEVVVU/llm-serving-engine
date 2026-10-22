import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import torch
from transformers import Qwen3ForCausalLM
from test_runtime import tiny_checkpoint, reference_tokens
from nanovllm import LLM, SamplingParams


def request(base, path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.status, json.load(response)


def offline(model_path, device):
    config = dict(device=device, kvcache_block_size=4, num_kvcache_blocks=24,
                  max_model_len=128, max_num_seqs=4, max_num_batched_tokens=64)
    prompts = [[3, 4, 5, 6, 7, 8], [8, 9, 10] * 3, [11, 12, 13] * 4 + [14]]
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    graph = LLM(model_path, **config)
    eager = None
    reference = None
    try:
        actual = graph.generate(prompts, params, False)
        eager = LLM(model_path, enforce_eager=True, **config)
        expected = eager.generate(prompts, params, False)
        assert [x["token_ids"] for x in actual] == [x["token_ids"] for x in expected]
        reference = Qwen3ForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16, attn_implementation="eager").to(device).eval()
        hf_tokens = [reference_tokens(reference, prompt, 8, device) for prompt in prompts]
        assert [x["token_ids"] for x in actual] == hf_tokens, (actual, hf_tokens)
        replay = prompts[0] + actual[0]["token_ids"][:6] + [21]
        assert graph.generate([replay], params, False)[0]["token_ids"] == eager.generate([replay], params, False)[0]["token_ids"]
        for seed in range(12):
            prompt = [22 + seed, 35 + seed] * 5
            got = graph.generate([prompt], params, False)[0]["token_ids"]
            want = eager.generate([prompt], params, False)[0]["token_ids"]
            assert got == want
        stats = graph.scheduler.block_manager.stats()
        assert stats["active_pages"] == 0 and stats["evictions"] > 0 and stats["prefix_hits"] > 0
        assert stats["total_pages"] == stats["cached_pages"] + stats["free_pages"]
        lanes = getattr(graph.model_runner, "replay_lanes", [])
        if device == "cuda":
            assert (3, 4) in lanes, lanes
            assert graph.model_runner.graph_replays > 0
        return dict(prompts=prompts, greedy_token_ids=hf_tokens, independent_hf_parity=True,
                    eager_graph_parity=True, pages=stats, graph_lanes=lanes,
                    graph_replays=getattr(graph.model_runner, "graph_replays", 0))
    finally:
        graph.exit()
        if eager is not None:
            eager.exit()
        del reference
        if device == "cuda":
            torch.cuda.empty_cache()


def http_drive(model, device, output):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_path = output.with_suffix(".server.log")
    log = log_path.open("w", encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")
    cmd = [sys.executable, "-u", "-m", "nanovllm.server", "--model", model,
           "--device", device, "--port", str(port), "--page-size", "4", "--pages", "40",
           "--max-model-len", "128", "--token-budget", "16", "--max-running", "4",
           "--client-buffer", "64", "--deadline", "30"]
    process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    evidence = dict(command=cmd, pid=process.pid, log=str(log_path))
    try:
        deadline = time.monotonic() + 90
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
            try:
                _, health = request(base, "/health")
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        evidence["doctor"] = health
        bodies = [dict(prompt=[3 + i, 4, 5] * (i + 1), max_tokens=6, temperature=0, ignore_eos=True) for i in range(3)]
        with ThreadPoolExecutor(3) as pool:
            responses = list(pool.map(lambda body: request(base, "/v1/completions", body), bodies))
        assert all(status == 200 and len(body["choices"][0]["token_ids"]) == 6 for status, body in responses)
        evidence["concurrent_requests"] = bodies
        evidence["concurrent_responses"] = responses
        stream_body = dict(prompt=[6, 7, 8, 9], max_tokens=6, temperature=0, ignore_eos=True, stream=True)
        req = urllib.request.Request(base + "/v1/completions", data=json.dumps(stream_body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            stream = response.read().decode()
        assert "data: [DONE]" in stream and '"done": true' in stream
        evidence["sse"] = stream
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        connection.request("POST", "/v1/completions", json.dumps(dict(prompt=[3, 4] * 4, max_tokens=100, temperature=0, ignore_eos=True, stream=True)), {"Content-Type": "application/json"})
        response = connection.getresponse()
        line = response.readline().decode()
        event = json.loads(line.removeprefix("data: "))
        status, cancel = request(base, "/requests/" + event["request_id"], method="DELETE")
        assert status == 202
        connection.close()
        evidence["cancel"] = dict(first_event=event, acknowledgement=cancel)
        for body, wanted in ((dict(prompt=[3], max_tokens=0), 200), (dict(prompt=[3] * 129, max_tokens=1), 400), (dict(prompt=[3], max_tokens=-1), 400)):
            try:
                status, value = request(base, "/v1/completions", body)
            except urllib.error.HTTPError as exc:
                status, value = exc.code, json.load(exc)
            assert status == wanted, (body, status, value)
        deadline = time.monotonic() + 15
        while True:
            _, stats = request(base, "/metrics")
            if stats.get("live_requests") == 0 and stats.get("pending_commands") == 0:
                break
            assert time.monotonic() < deadline, stats
            time.sleep(0.02)
        assert stats["active_pages"] == 0, stats
        assert stats["accepted"] == stats["completed"] + stats["cancelled"] + stats["failed"], stats
        assert stats["cancelled"] >= 1, stats
        assert stats["total_pages"] == stats["free_pages"] + stats["cached_pages"], stats
        evidence["final_metrics"] = stats
        return evidence
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()
        evidence["cleanup"] = dict(child_exited=process.poll() is not None, returncode=process.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("evidence/runtime.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(device=args.device, torch=torch.__version__, cuda=torch.version.cuda)
    if args.device == "cuda":
        evidence["gpu"] = torch.cuda.get_device_name()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            model = args.model or temporary
            if not args.model:
                tiny_checkpoint(model)
            evidence["model"] = str(model) if args.model else "generated deterministic tiny Qwen3"
            evidence["offline"] = offline(model, args.device)
            evidence["http"] = http_drive(model, args.device, args.output)
        evidence["passed"] = True
    except Exception as exc:
        evidence["passed"] = False
        evidence["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    assert args.output.is_file()
    print(json.dumps(dict(passed=True, evidence=str(args.output))))


if __name__ == "__main__":
    main()
