import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    backend: str = "torch"
    device: str = "cpu"
    compile: bool = False
    max_pending_requests: int = 64

    def __post_init__(self):
        capacities = (self.kvcache_block_size, self.max_num_seqs, self.max_num_batched_tokens,
                      self.max_model_len, self.max_pending_requests, self.tensor_parallel_size,
                      self.num_kvcache_blocks)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in capacities):
            raise ValueError("capacities must be integers")
        if not os.path.isdir(self.model):
            raise ValueError("model must be a local safetensors directory")
        if self.backend not in ("torch", "native") or self.device not in ("cpu", "cuda"):
            raise ValueError("backend must be torch/native and device cpu/cuda")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("invalid tensor parallel size")
        if self.backend == "torch" and self.tensor_parallel_size != 1:
            raise ValueError("portable backend supports tensor_parallel_size=1")
        if min(self.kvcache_block_size, self.max_num_seqs, self.max_num_batched_tokens,
               self.max_model_len, self.max_pending_requests) <= 0:
            raise ValueError("capacities must be positive")
        if self.backend == "native" and (self.device != "cuda" or self.kvcache_block_size % 256):
            raise ValueError("native backend requires cuda and block size divisible by 256")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("invalid GPU memory utilization")
        if self.num_kvcache_blocks == -1 and self.backend == "torch":
            self.num_kvcache_blocks = 64
        if self.backend == "torch" and self.num_kvcache_blocks <= 0:
            raise ValueError("portable page pool must be positive")
        self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True, trust_remote_code=False)
        self.hf_config.dtype = getattr(self.hf_config, "dtype", None) or getattr(self.hf_config, "torch_dtype", None) or torch.float32
        if self.hf_config.model_type != "qwen3" or getattr(self.hf_config, "sliding_window", None):
            raise ValueError("only full-attention Qwen3 is supported")
        rope = getattr(self.hf_config, "rope_parameters", None) or getattr(self.hf_config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("only default RoPE is supported")
        self.hf_config.rope_theta = rope.get("rope_theta", getattr(self.hf_config, "rope_theta", 1000000))
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
