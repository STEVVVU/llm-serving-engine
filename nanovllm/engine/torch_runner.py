import torch
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, get_context, reset_context


class TorchRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        dtype = torch.float32 if self.device.type == "cpu" else config.hf_config.dtype
        dtype = dtype or torch.bfloat16
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device(self.device):
                self.model = Qwen3ForCausalLM(config.hf_config).eval()
        finally:
            torch.set_default_dtype(original_dtype)
        with torch.device(self.device):
            load_model(self.model, config.model)
            self.sampler = Sampler()
            self.block_size = config.kvcache_block_size
            self.scratch_page = config.num_kvcache_blocks
            scratch_pages = (config.max_num_seqs + self.block_size - 1) // self.block_size
            h = config.hf_config
            self.kv_cache = torch.zeros(2, h.num_hidden_layers,
                config.num_kvcache_blocks + scratch_pages, self.block_size,
                h.num_key_value_heads, h.head_dim, dtype=dtype)
            for i, layer in enumerate(self.model.model.layers):
                layer.self_attn.attn.k_cache = self.kv_cache[0, i]
                layer.self_attn.attn.v_cache = self.kv_cache[1, i]
        if config.compile:
            for module in self.model.modules():
                if module.__class__.__name__ in ("RMSNorm", "RotaryEmbedding", "SiluAndMul"):
                    module.forward = torch.compile(module.forward)
            self.sampler.forward = torch.compile(self.sampler.forward)
        self.graphs = {}
        self.graph_replays = 0
        self.replay_lanes = set()
        if self.device.type == "cuda" and not config.enforce_eager:
            self.capture_cudagraph()

    def tensor(self, data, dtype=torch.int64):
        return torch.tensor(data, dtype=dtype, device=self.device)

    def prepare(self, seqs, prefill):
        ids, positions, slots, qends, kends, lengths = [], [], [], [0], [0], []
        width = (self.config.max_model_len + self.block_size - 1) // self.block_size
        tables = []
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            ids.extend(seq[start:end])
            positions.extend(range(start, end))
            slots.extend(seq.block_table[p // self.block_size] * self.block_size + p % self.block_size for p in range(start, end))
            qends.append(len(ids))
            kends.append(kends[-1] + end)
            lengths.append(end)
            tables.append(seq.block_table + [self.scratch_page] * (width - len(seq.block_table)))
        set_context(prefill, self.tensor(qends), self.tensor(kends), max(s.num_scheduled_tokens for s in seqs),
                    max(lengths), self.tensor(slots), self.tensor(lengths), self.tensor(tables))
        return self.tensor(ids), self.tensor(positions)

    @torch.inference_mode()
    def logits(self, seqs, prefill):
        ids, positions = self.prepare(seqs, prefill)
        try:
            if not prefill and self.graphs:
                bs = len(seqs)
                bucket = next(x for x in self.graph_bs if x >= bs)
                state = self.graph_vars
                state["ids"].zero_()
                state["positions"].zero_()
                state["lengths"].zero_()
                state["tables"].fill_(self.scratch_page)
                state["slots"].copy_(self.scratch_slots)
                ctx = get_context()
                state["ids"][:bs].copy_(ids)
                state["positions"][:bs].copy_(positions)
                state["lengths"][:bs].copy_(ctx.context_lens)
                state["tables"][:bs].copy_(ctx.block_tables)
                state["slots"][:bs].copy_(ctx.slot_mapping)
                self.graphs[bucket].replay()
                self.graph_replays += 1
                self.replay_lanes.add((bs, bucket))
                return self.model.compute_logits(state["outputs"][:bs])
            return self.model.compute_logits(self.model(ids, positions))
        finally:
            reset_context()

    def run(self, seqs, prefill):
        logits = self.logits(seqs, prefill)
        return self.sampler(logits, self.tensor([s.temperature for s in seqs], torch.float32)).tolist()

    @torch.inference_mode()
    def capture_cudagraph(self):
        n = min(self.config.max_num_seqs, self.config.num_kvcache_blocks, self.config.max_num_batched_tokens)
        self.graph_bs = sorted(set([x for x in (1, 2, 4, 8) if x <= n] + list(range(16, n + 1, 16)) + [n]))
        width = (self.config.max_model_len + self.block_size - 1) // self.block_size
        self.scratch_slots = torch.arange(n, device=self.device) + self.scratch_page * self.block_size
        state = dict(ids=self.tensor([0] * n), positions=self.tensor([0] * n),
                     lengths=self.tensor([0] * n), slots=self.scratch_slots.clone(),
                     tables=torch.full((n, width), self.scratch_page, device=self.device, dtype=torch.int64),
                     outputs=torch.empty((n, self.config.hf_config.hidden_size), device=self.device, dtype=next(self.model.parameters()).dtype))
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        pool = None
        for bs in reversed(self.graph_bs):
            set_context(False, slot_mapping=state["slots"][:bs], context_lens=state["lengths"][:bs], block_tables=state["tables"][:bs])
            with torch.cuda.stream(stream):
                for _ in range(3):
                    state["outputs"][:bs].copy_(self.model(state["ids"][:bs], state["positions"][:bs]))
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                state["outputs"][:bs].copy_(self.model(state["ids"][:bs], state["positions"][:bs]))
            pool = graph.pool()
            self.graphs[bs] = graph
        torch.cuda.synchronize(self.device)
        reset_context()
        self.graph_vars = state

    def call(self, name, *args):
        return getattr(self, name)(*args)

    def exit(self):
        self.graphs.clear()
        self.model = None
        self.kv_cache = None
        if self.device.type == "cuda":
            self.graph_vars = None
            torch.cuda.empty_cache()
