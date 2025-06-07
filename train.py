import argparse
import time
from functools import cache
import os
from dataset import WenwuDataset
from torch.utils.data import DataLoader
from cn_clip.clip import load_from_name, tokenize
import torch
from torch import LongTensor, nn
import tqdm

from datetime import datetime

torch.backends.verbose = True
from model import WenWuClip
from train_helpers import train, evaluate, save_state, read_state, read_reports, save_reports, tackle, move_to
from freeze_strategies import freeze

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=6)
parser.add_argument("-l", "--lr", type=float, default=5e-7)
parser.add_argument("-c", "--checkpoint", type=str, default=None)
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

use_amp = (args.precision == "amp") and (device == "cuda")
scaler = torch.GradScaler(device=device, enabled=use_amp)

model, preprocess = load_from_name(args.base, device=device, download_root='./base')

train_dataset = WenwuDataset(0, 0 + (0.8 * args.data_scale), args.image_in_memory, preprocess)
train_loader = DataLoader(train_dataset, num_workers=args.workers,
                          persistent_workers=True,
                          prefetch_factor=4,
                          shuffle=True, batch_size=args.batch_size,
                          pin_memory=True)

val_dataset = WenwuDataset(0.8, 0.8 + (0.1 * args.data_scale), args.image_in_memory, preprocess)
val_loader = DataLoader(val_dataset, num_workers=args.workers,
                        persistent_workers=True,
                        shuffle=True, batch_size=args.batch_size,
                        pin_memory=True)

criterion_img = nn.CrossEntropyLoss().to(device)
criterion_text = nn.CrossEntropyLoss().to(device)


def freeze_and_get_optimizer(_optimizer_state=None):
    freeze(model, args.freeze_mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    _optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    if _optimizer_state is not None:
        _optimizer.load_state_dict(_optimizer_state)
    return _optimizer


if args.checkpoint is not None:
    model_state, optimizer_state, scaler_state, start_epoch = read_state(args.checkpoint, device)
    scaler.load_state_dict(scaler_state)
    model.load_state_dict(model_state)
    optimizer = freeze_and_get_optimizer(optimizer_state)
    reports = read_reports(args.project)
else:
    optimizer = freeze_and_get_optimizer()
    start_epoch = 1
    lowest_loss = float("inf")
    reports = {
        "history": []
    }

# optimizer

tackle(model)
move_to(model, device, args.precision)

os.makedirs("checkpoints", exist_ok=True)
print(f"use_amp={use_amp} device={device} precision={args.precision}")
for epoch in range(start_epoch, args.epochs + 1):
    train(f"[{args.project}]{epoch}/{args.epochs}", model, train_loader, optimizer,
          device, args.precision, use_amp,
          scaler=scaler,
          criterion_img=criterion_img, criterion_text=criterion_text)
    accuracy, val_loss = evaluate(model, val_loader, device)

    reports["history"].append({
        "epoch": epoch,
        "val_accuracy": accuracy,
        "val_loss": val_loss.item(),
    })
    save_reports(args.project, reports)

    if val_loss < lowest_loss:
        lowest_loss = val_loss
        save_state(filename=f"checkpoints/{args.project}.pt",
                   model=model, optimizer=optimizer, epoch=epoch, scaler=scaler)

if val_loss < lowest_loss:
    lowest_loss = val_loss
    save_state(filename=f"checkpoints/{args.project}.pt",
               model=model, optimizer=optimizer, epoch=epoch, scaler=scaler)
