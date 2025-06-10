import argparse
import os
from pathlib import Path

import torch
from cn_clip.clip import load_from_name
from cn_clip.clip.model import CLIP
from torch.utils.data import DataLoader

from dataset import WenwuDataset

torch.backends.verbose = True
torch.backends.cudnn.benchmark = True
from train_helpers import train, evaluate, save_state, read_state, read_reports, save_reports, tackle, move_to, \
    evaluate_clip_multicap
from freeze_strategies import freeze

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-l", "--lr", type=float, default=5e-7)
parser.add_argument("--image-in-memory", type=bool, default=False)
parser.add_argument("-w", "--workers", type=int, default=8)
parser.add_argument("-d", "--device", type=str, default=None)
parser.add_argument("--base", type=str, default="ViT-H-14")
parser.add_argument("--data-scale", type=float, default=1)
parser.add_argument("--freeze-mode", type=str, default="a")
parser.add_argument("-p", "--project", type=str, default="default")
parser.add_argument(
    "--precision",
    choices=["fp32", "amp", "fp16"],  # 新增 'fp32' / 'fp16'
    default="fp32"
)

parser.add_argument("--save-interval-epochs", type=int, default=1)
args = parser.parse_args()

if args.image_in_memory:
    print("Using Image In-Memory")

if args.device is not None:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

use_amp = args.precision == "amp"
scaler = torch.GradScaler(device=device, enabled=use_amp)

train_dataset = WenwuDataset(0, 0 + (0.8 * args.data_scale), args.image_in_memory, device=device)
train_loader = DataLoader(train_dataset, num_workers=args.workers,
                          persistent_workers=args.workers > 0,
                          prefetch_factor=4 if args.workers > 0 else None,
                          shuffle=True, batch_size=args.batch_size,
                          pin_memory=True)

val_dataset = WenwuDataset(0.8, 0.8 + (0.1 * args.data_scale), args.image_in_memory, device=device)
val_loader = DataLoader(val_dataset, num_workers=args.workers,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None,
                        shuffle=True, batch_size=args.batch_size,
                        pin_memory=True)


def freeze_and_get_optimizer(_model: CLIP, _optimizer_state=None):
    freeze(_model, args.freeze_mode)
    trainable_params = [p for p in _model.parameters() if p.requires_grad]
    _optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    if _optimizer_state is not None:
        _optimizer.load_state_dict(_optimizer_state)
    return _optimizer


if __name__ == "__main__":
    model, preprocess = load_from_name(args.base, device=device, download_root='./base')

    model = move_to(model, device, args.precision)
    checkpoint = Path("checkpoints") / (args.project + ".pt")
    if args.project is not None and checkpoint.exists():
        model_state, optimizer_state, scaler_state, start_epoch = read_state(checkpoint, device)
        scaler.load_state_dict(scaler_state)
        model.load_state_dict(model_state)
        optimizer = freeze_and_get_optimizer(model, optimizer_state)

    else:
        optimizer = freeze_and_get_optimizer(model)
        start_epoch = 1

    reports = read_reports(args.project)
    print(f"use_amp={use_amp} device={device} precision={args.precision}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(f"[{args.project}]{epoch}/{args.epochs} Training",
                           model=model,
                           train_loader=train_loader,
                           optimizer=optimizer,
                           device=device, precision=args.precision, use_amp=use_amp,
                           scaler=scaler)

        eval_result = evaluate_clip_multicap(model, val_loader, device)

        ## 一些打印和保存结果的工作
        score = eval_result["score"]
        highest_score = 0
        for hitem in reports["history"]:
            s = hitem["performance"]["score"]
            if s > highest_score:
                highest_score = s

        reports["history"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "performance": eval_result,
        })
        print(eval_result)
        save_reports(args.project, reports)


        if score > highest_score:
            highest_score = score
            save_state(filename=f"checkpoints/{args.project}.pt",
                       model=model, optimizer=optimizer, epoch=epoch, scaler=scaler)
