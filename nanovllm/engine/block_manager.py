from collections import deque
from dataclasses import dataclass, field
from hashlib import blake2b

from nanovllm.engine.sequence import Sequence


@dataclass(eq=False)
class RadixNode:
    parent: "RadixNode | None" = None
    tokens: tuple[int, ...] = ()
    block_id: int = -1
    children: dict = field(default_factory=dict)
    touched: int = 0


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.node = None


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
        self.root = RadixNode()
        self.clock = self.hits = self.evictions = 0

    @classmethod
    def compute_hash(cls, token_ids, prefix=-1):
        return int.from_bytes(blake2b(repr((prefix, token_ids)).encode(), digest_size=8).digest(), "little")

    def _touch(self, node):
        self.clock += 1
        node.touched = self.clock

    def _allocate_block(self):
        if not self.free_block_ids:
            leaves = [b.node for b in self.blocks if b.ref_count == 0 and b.node is not None and not b.node.children]
            if not leaves:
                raise RuntimeError("no evictable KV page")
            node = min(leaves, key=lambda n: n.touched)
            del node.parent.children[node.tokens]
            self.blocks[node.block_id].node = None
            self.free_block_ids.append(node.block_id)
            self.evictions += 1
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0 and block.node is None
        block.ref_count = 1
        self.used_block_ids.add(block_id)
        return block_id

    def _release(self, block_id):
        block = self.blocks[block_id]
        assert block.ref_count > 0
        block.ref_count -= 1
        if not block.ref_count:
            self.used_block_ids.remove(block_id)
            if block.node is None:
                self.free_block_ids.append(block_id)
            else:
                self._touch(block.node)

    def _prefix(self, seq):
        node = self.root
        matches = []
        # Keep one prompt token to compute next-token logits.
        for i in range((len(seq) - 1) // self.block_size):
            node = node.children.get(tuple(seq.block(i)))
            if node is None:
                break
            matches.append(node)
        return matches

    def can_allocate(self, seq: Sequence):
        matches = self._prefix(seq)
        available = sum(b.ref_count == 0 for b in self.blocks)
        needed = seq.num_blocks - sum(self.blocks[n.block_id].ref_count > 0 for n in matches)
        return len(matches) if available >= needed else -1

    def allocate(self, seq, num_cached_blocks):
        assert not seq.block_table
        for node in self._prefix(seq)[:num_cached_blocks]:
            block = self.blocks[node.block_id]
            block.ref_count += 1
            self.used_block_ids.add(block.block_id)
            self._touch(node)
            seq.block_table.append(block.block_id)
            self.hits += 1
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq):
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def can_append(self, seq):
        return len(seq.block_table) >= seq.num_blocks or any(b.ref_count == 0 for b in self.blocks)

    def may_append(self, seq):
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq):
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        node = self.root
        for i in range(end):
            tokens = tuple(seq.block(i))
            block = self.blocks[seq.block_table[i]]
            child = node.children.get(tokens)
            if child is None:
                child = RadixNode(node, tokens, block.block_id)
                node.children[tokens] = child
                block.node = child
            elif child.block_id != block.block_id:
                self.blocks[child.block_id].ref_count += 1
                self.used_block_ids.add(child.block_id)
                self._release(block.block_id)
                seq.block_table[i] = child.block_id
            self._touch(child)
            node = child

    def stats(self):
        active = sum(b.ref_count > 0 for b in self.blocks)
        cached = sum(b.ref_count == 0 and b.node is not None for b in self.blocks)
        return dict(total_pages=len(self.blocks), active_pages=active, cached_pages=cached,
                    free_pages=len(self.free_block_ids), prefix_hits=self.hits, evictions=self.evictions)
