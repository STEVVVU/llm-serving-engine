from collections import deque
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.max_pending_requests = config.max_pending_requests
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, self.block_size)
        self.waiting, self.running, self.completed = deque(), deque(), deque()
        self.active = {}
        self.prefer_prefill = True

    def is_finished(self):
        return not self.waiting and not self.running and not self.completed

    def add(self, seq):
        if len(self.waiting) + len(self.running) + len(self.completed) >= self.max_pending_requests:
            raise ValueError("request queue is full")
        total = len(seq) + max(0, seq.max_tokens - 1)
        pages = (total + self.block_size - 1) // self.block_size
        if total > self.max_model_len or pages > len(self.block_manager.blocks):
            raise ValueError("request exceeds model context or KV pool capacity")
        if seq.max_tokens == 0:
            seq.status = SequenceStatus.FINISHED
            self.completed.append(seq)
        else:
            self.waiting.append(seq)

    def _admit(self, seq):
        if seq.seq_id in self.active:
            return True
        if len(self.active) >= self.max_num_seqs:
            return False
        pages = (len(seq) + seq.max_tokens - 2 + self.block_size) // self.block_size
        if sum(self.active.values()) + pages > len(self.block_manager.blocks):
            return False
        cached = self.block_manager.can_allocate(seq)
        if cached < 0:
            return False
        self.block_manager.allocate(seq, cached)
        self.active[seq.seq_id] = pages
        return True

    def schedule(self):
        prefill = bool(self.waiting) and (self.prefer_prefill or not self.running)
        scheduled = []
        budget = self.max_num_batched_tokens
        if prefill:
            for _ in range(len(self.waiting)):
                seq = self.waiting.popleft()
                if budget and len(scheduled) < self.max_num_seqs and self._admit(seq):
                    seq.num_scheduled_tokens = min(len(seq) - seq.num_cached_tokens, budget)
                    budget -= seq.num_scheduled_tokens
                    scheduled.append(seq)
                self.waiting.append(seq)
        if not scheduled:
            prefill = False
            for _ in range(min(len(self.running), self.max_num_seqs, budget)):
                seq = self.running.popleft()
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                scheduled.append(seq)
                self.running.append(seq)
        self.prefer_prefill = not prefill
        return scheduled, prefill

    def cancel(self, seq_id):
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    self.block_manager.deallocate(seq)
                    self.active.pop(seq_id, None)
                    seq.status = SequenceStatus.FINISHED
                    return True
        return False

    def postprocess(self, seqs, token_ids, is_prefill):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < len(seq):
                continue
            if is_prefill:
                self.waiting.remove(seq)
                self.running.append(seq)
                seq.status = SequenceStatus.RUNNING
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.active.pop(seq.seq_id)
                self.running.remove(seq)
