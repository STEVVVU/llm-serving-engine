import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.closed = False
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        if config.backend == "native":
            from nanovllm.engine.model_runner import ModelRunner
        else:
            from nanovllm.engine.torch_runner import TorchRunner
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) if config.backend == "native" else TorchRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, local_files_only=True, trust_remote_code=False)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self.closed:
            return
        self.closed = True
        for seq in list(self.scheduler.waiting) + list(self.scheduler.running):
            self.scheduler.cancel(seq.seq_id)
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not isinstance(prompt, list) or not prompt or any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.config.hf_config.vocab_size for t in prompt):
            raise ValueError("prompt must contain valid token IDs")
        seq = Sequence(prompt, sampling_params, self.config.kvcache_block_size)
        self.scheduler.add(seq)
        return seq.seq_id

    def cancel_request(self, seq_id):
        return self.scheduler.cancel(seq_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        completed = list(self.scheduler.completed)
        self.scheduler.completed.clear()
        if not seqs:
            return [(seq.seq_id, seq.completion_token_ids) for seq in completed], 0
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs + completed if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("one sampling parameter set is required per prompt")
        pending = iter(zip(prompts, sampling_params))
        exhausted = False
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not exhausted or not self.is_finished():
            while not exhausted and len(self.scheduler.waiting) + len(self.scheduler.running) + len(self.scheduler.completed) < self.config.max_pending_requests:
                try:
                    prompt, sp = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                self.add_request(prompt, sp)
            if self.is_finished():
                break
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
