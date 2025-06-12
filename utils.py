import torch


def determine_device(cmd_args: any = None, ddp_rank: any = None):
    if cmd_args.device is not None:
        return cmd_args.device
    elif torch.cuda.is_available():
        if ddp_rank is not None:
            return torch.device("cuda", ddp_rank)
        else:
            return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
