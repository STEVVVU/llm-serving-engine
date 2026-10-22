import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1
