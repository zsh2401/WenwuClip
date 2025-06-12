# dist_utils.py
import os, torch.distributed as dist, torch

def setup_distributed():
    """
    若脚本以 torchrun 启动则初始化进程组，
    否则返回 world_size=1, rank=0 (单机/单卡)
    """
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        distributed = True
        # print(f"Running under ddp {world_size} {}")
    else:
        local_rank = 0
        world_size = 1
        distributed = False
    return distributed, local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()