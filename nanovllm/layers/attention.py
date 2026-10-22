import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.context import get_context


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.scale, self.num_kv_heads = scale, num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.backend = "torch"

    def forward(self, q, k, v):
        if self.backend == "native":
            from nanovllm.layers.native_attention import Attention as NativeAttention
            return NativeAttention.forward(self, q, k, v)
        ctx = get_context()
        shape = (-1, self.num_kv_heads, self.head_dim)
        kc, vc = self.k_cache.view(shape), self.v_cache.view(shape)
        # Padding lanes write only to reserved scratch slots, never a live page.
        kc.index_copy_(0, ctx.slot_mapping.long(), k)
        vc.index_copy_(0, ctx.slot_mapping.long(), v)
        if ctx.is_prefill:
            out = []
            qends, kends = ctx.cu_seqlens_q.tolist(), ctx.cu_seqlens_k.tolist()
            for i, (start, end) in enumerate(zip(qends, qends[1:])):
                length = kends[i + 1] - kends[i]
                pos = torch.arange(length, device=q.device)
                slots = ctx.block_tables[i, pos // self.k_cache.size(1)].long() * self.k_cache.size(1) + pos % self.k_cache.size(1)
                keys, vals = kc[slots], vc[slots]
                mask = pos[None, :] <= torch.arange(length - (end - start), length, device=q.device)[:, None]
                out.append(self.attend(q[start:end].transpose(0, 1), keys.transpose(0, 1), vals.transpose(0, 1), mask).transpose(0, 1))
            return torch.cat(out)
        pos = torch.arange(ctx.block_tables.size(1) * self.k_cache.size(1), device=q.device)
        slots = ctx.block_tables[:, pos // self.k_cache.size(1)].long() * self.k_cache.size(1) + pos % self.k_cache.size(1)
        keys, vals = kc[slots], vc[slots]
        mask = (pos[None, :] < ctx.context_lens[:, None])[:, None, None, :]
        return self.attend(q.unsqueeze(2), keys.transpose(1, 2), vals.transpose(1, 2), mask).squeeze(2)

    def attend(self, q, k, v, mask):
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=-3)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=-3)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
