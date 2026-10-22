import json
from pathlib import Path
import tempfile
import unittest
import queue
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.utils.loader import load_model


def tiny_checkpoint(path):
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, tie_word_embeddings=True,
        eos_token_id=2, pad_token_id=0)
    model = Qwen3ForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", eos_token="t2", pad_token="t0").save_pretrained(path)
    return model


def reference_tokens(model, prompt, count, device="cpu"):
    ids = list(prompt)
    with torch.inference_mode():
        for _ in range(count):
            logits = model(torch.tensor([ids], device=device)).logits[0, -1]
            ids.append(logits.argmax().item())
    return ids[len(prompt):]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = tiny_checkpoint(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def engine(self, **kwargs):
        settings = dict(kvcache_block_size=4, num_kvcache_blocks=32,
                        max_model_len=64, max_num_seqs=4, max_num_batched_tokens=5)
        settings.update(kwargs)
        engine = LLM(self.tmp.name, **settings)
        self.addCleanup(engine.exit)
        return engine

    def assert_conserved(self, manager):
        stats = manager.stats()
        self.assertEqual(stats["total_pages"], stats["active_pages"] + stats["cached_pages"] + stats["free_pages"])

    def test_packed_chunk_logits_and_greedy(self):
        engine = self.engine()
        prompts = [[3, 4, 5], [6, 7, 8, 9, 10, 11, 12], [13] * 12]
        for prompt in prompts:
            engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=5, ignore_eos=True))
        finished, steps = {}, 0
        while not engine.is_finished():
            seqs, prefill = engine.scheduler.schedule()
            self.assertTrue(seqs)
            self.assertLessEqual(sum(s.num_scheduled_tokens for s in seqs), 5)
            logits = engine.model_runner.logits(seqs, prefill)
            for i, seq in enumerate(seqs):
                end = seq.num_cached_tokens + seq.num_scheduled_tokens
                with torch.inference_mode():
                    expected = self.model(torch.tensor([seq.token_ids[:end]])).logits[0, -1]
                torch.testing.assert_close(logits[i], expected, atol=2e-5, rtol=2e-4)
            engine.scheduler.postprocess(seqs, logits.argmax(-1).tolist(), prefill)
            for seq in seqs:
                if seq.is_finished:
                    finished[seq.seq_id] = seq.completion_token_ids
            steps += 1
            self.assertLess(steps, 100)
            self.assert_conserved(engine.scheduler.block_manager)
        self.assertEqual(list(finished.values()), [reference_tokens(self.model, p, 5) for p in prompts])
        self.assertEqual(engine.scheduler.block_manager.stats()["active_pages"], 0)

    def test_prefix_including_generated_pages(self):
        engine = self.engine(max_num_batched_tokens=8)
        prompt = [3, 4, 5, 6, 7]
        result = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), False)[0]
        replay = prompt + result["token_ids"][:7] + [9]
        ids = engine.generate([replay], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True), False)[0]["token_ids"]
        self.assertEqual(ids, reference_tokens(self.model, replay, 3))
        self.assertGreaterEqual(engine.scheduler.block_manager.hits, 3)
        self.assert_conserved(engine.scheduler.block_manager)

    def test_exact_radix_collision_pins_and_eviction(self):
        manager = BlockManager(5, 2)
        def process(tokens):
            seq = Sequence(tokens, block_size=2)
            manager.allocate(seq, manager.can_allocate(seq))
            seq.num_scheduled_tokens = len(seq)
            manager.hash_blocks(seq)
            manager.deallocate(seq)
            return seq
        with patch.object(BlockManager, "compute_hash", return_value=1):
            process([1, 2, 3, 4])
            process([8, 9, 3, 4])
            wrong = Sequence([1, 9, 3, 4, 5], block_size=2)
            self.assertEqual(manager.can_allocate(wrong), 0)
            pinned = Sequence([1, 2, 3, 4, 5], block_size=2)
            manager.allocate(pinned, manager.can_allocate(pinned))
            original = pinned.block_table[:2]
            other = process([20, 21, 22, 23])
            self.assertEqual(pinned.block_table[:2], original)
            self.assertTrue(all(manager.blocks[b].ref_count == 1 for b in original))
            self.assertGreater(manager.evictions, 0)
            manager.deallocate(pinned)
            self.assert_conserved(manager)

    def test_budget_fairness_admission_cancel_and_zero(self):
        engine = self.engine(max_num_seqs=1, max_num_batched_tokens=1)
        first = engine.add_request([3], SamplingParams(temperature=0, max_tokens=5, ignore_eos=True))
        engine.step()
        for _ in range(3):
            engine.add_request([4], SamplingParams(temperature=0, max_tokens=2, ignore_eos=True))
        for _ in range(4):
            engine.step()
            self.assertLessEqual(len(engine.scheduler.active), 1)
        self.assertNotIn(first, engine.scheduler.active)
        for seq in list(engine.scheduler.waiting) + list(engine.scheduler.running):
            engine.cancel_request(seq.seq_id)
        self.assertTrue(engine.is_finished())
        zero = engine.generate([[3]], SamplingParams(max_tokens=0), False)
        self.assertEqual(zero[0]["token_ids"], [])
        with self.assertRaises(ValueError):
            engine.add_request([3] * 65, SamplingParams(max_tokens=1))
        self.assertEqual(engine.scheduler.block_manager.stats()["active_pages"], 0)

    def test_decode_rotation_with_budget(self):
        engine = self.engine(max_num_batched_tokens=3)
        for _ in range(3):
            engine.add_request([3], SamplingParams(max_tokens=5, ignore_eos=True))
        engine.step()
        engine.scheduler.max_num_batched_tokens = 1
        counts = {s.seq_id: s.num_completion_tokens for s in engine.scheduler.running}
        for _ in range(3):
            seqs, prefill = engine.scheduler.schedule()
            self.assertEqual(len(seqs), 1)
            self.assertFalse(prefill)
            engine.scheduler.postprocess(seqs, [4], False)
        self.assertTrue(all(s.num_completion_tokens == counts[s.seq_id] + 1 for s in engine.scheduler.running))

    def test_offline_generate_refills_bounded_queue(self):
        engine = self.engine(max_pending_requests=2)
        outputs = engine.generate([[3]] * 7, SamplingParams(temperature=0, max_tokens=1), False)
        self.assertEqual(len(outputs), 7)
        self.assertTrue(engine.is_finished())

    def test_float32_norm_preserves_residual_and_input(self):
        norm = RMSNorm(4)
        x, residual = torch.randn(3, 4), torch.randn(3, 4)
        original_x, original_r = x.clone(), residual.clone()
        output, summed = norm(x, residual)
        torch.testing.assert_close(summed, original_x + original_r)
        expected = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + norm.eps)
        torch.testing.assert_close(output, expected)
        norm(x)
        torch.testing.assert_close(x, original_x)
        torch.testing.assert_close(residual, original_r)

    def test_slow_client_backpressure_does_not_block_owner(self):
        from nanovllm.server import Service, Client
        engine = self.engine()
        service = Service(engine, buffer_tokens=1)
        try:
            slow = Client(dict(prompt=[3, 4], max_tokens=20, temperature=0, ignore_eos=True), queue.Queue(1))
            service.commands.put(slow)
            fast = Client(dict(prompt=[5, 6], max_tokens=3, temperature=0, ignore_eos=True), queue.Queue(16))
            service.commands.put(fast)
            while True:
                event = fast.events.get(timeout=5)
                self.assertNotIn("error", event)
                if event.get("done"):
                    self.assertEqual(len(event["token_ids"]), 3)
                    break
            rejected = slow.events.get(timeout=5)
            self.assertEqual(rejected.get("status"), 429)
        finally:
            service.close()
        self.assertEqual(service.counts, dict(accepted=2, completed=1, cancelled=1, failed=0))
        self.assertEqual(engine.scheduler.block_manager.stats()["active_pages"], 0)

    def test_terminal_queue_consumer_race(self):
        from nanovllm.server import Service, Client
        class ConsumerRaceQueue(queue.Queue):
            def put_nowait(self, item):
                if self.full():
                    self.get_nowait()
                    raise queue.Full
                return super().put_nowait(item)
        engine = self.engine()
        service = Service.__new__(Service)
        service.engine = engine
        service.counts = dict(accepted=1, completed=0, cancelled=0, failed=0)
        events = ConsumerRaceQueue(1)
        events.put_nowait(dict(token_ids=[4]))
        client = Client({}, events)
        client.seq_id = engine.add_request([3], SamplingParams(max_tokens=4))
        service.clients = {client.seq_id: client}
        service.terminal(client, dict(done=True), "completed")
        self.assertEqual(events.get_nowait(), dict(done=True))
        self.assertEqual(service.counts["completed"], 1)
        self.assertFalse(service.clients)
        self.assertTrue(engine.is_finished())

    def test_cli_rejects_public_binding(self):
        result = subprocess.run([sys.executable, "-m", "nanovllm.server", "--model", self.tmp.name, "--host", "0.0.0.0"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("numeric IPv4 loopback", result.stderr)

    def test_instance_pages_validation_and_loader(self):
        legacy = SimpleNamespace(model_type="qwen3", max_position_embeddings=128, torch_dtype=torch.float16)
        with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=legacy):
            config = Config(self.tmp.name)
        self.assertEqual(config.hf_config.dtype, torch.float16)
        a = self.engine(kvcache_block_size=2)
        b = self.engine(kvcache_block_size=8)
        a.add_request([3, 4, 5], SamplingParams(max_tokens=1))
        b.add_request([3, 4, 5], SamplingParams(max_tokens=1))
        self.assertEqual(a.scheduler.waiting[0].block_size, 2)
        self.assertEqual(b.scheduler.waiting[0].block_size, 8)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, "no safetensors"):
                load_model(a.model_runner.model, empty)
            from safetensors.torch import save_file
            save_file({"model.norm.weight": torch.ones(32)}, str(Path(empty) / "partial.safetensors"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_model(a.model_runner.model, empty)
        for params in (dict(max_tokens=-1), dict(temperature=float("nan")), dict(temperature=-1), dict(ignore_eos="yes")):
            with self.assertRaises(ValueError):
                SamplingParams(**params)


if __name__ == "__main__":
    unittest.main()
