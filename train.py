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
from train_helpers import freeze, train, evaluate, save_state, read_state, read_reports, save_reports, tackle, move_to

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=6)
parser.add_argument("-l", "--lr", type=float, default=5e-7)
parser.add_argument("-c", "--checkpoint", type=str, default=None)
parser.add_argument("-d", "--device", type=str, default=None)
parser.add_argument("--base", type=str, default="ViT-H-14")
parser.add_argument("--data-scale", type=float, default=1)
parser.add_argument("-p", "--project", type=str, default="default")
parser.add_argument(
    "--precision",
    choices=["fp32", "amp", "fp16"],   # 新增 'fp32' / 'fp16'
    default="fp32"
)
parser.add_argument("--save-interval-epochs", type=int, default=1)
args = parser.parse_args()

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

train_dataset = WenwuDataset(0, 0 + (0.8 * args.data_scale))
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)

val_dataset = WenwuDataset(0.8, 0.8 + (0.1 * args.data_scale))
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)

criterion_img = nn.CrossEntropyLoss().to(device)
criterion_text = nn.CrossEntropyLoss().to(device)

if args.checkpoint is not None:
    model, optimizer, start_epoch, lowest_loss = read_state(args.checkpoint)
    reports = read_reports(args.project)
else:
    model, preprocess = load_from_name(args.base, device=device, download_root='./base')
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    start_epoch = 1
    lowest_loss = float("inf")
    reports = {
        "history": []
    }

freeze(model)
tackle(model)
move_to(model,device,args.precision)
os.makedirs("checkpoints", exist_ok=True)
print(f"use_amp={use_amp} device={device} precision={args.precision}")
for epoch in range(start_epoch, args.epochs + 1):
    train(f"[{args.project}]{epoch}/{args.epochs}", model, train_loader, optimizer, device,args.precision, use_amp, scaler,
          criterion_img, criterion_text)
    accuracy, val_loss = evaluate(model, val_loader, device)

    reports["history"].append({
        "epoch": epoch,
        "val_accuracy": accuracy,
        "val_loss": val_loss.item(),
    })
    save_reports(args.project, reports)

    if val_loss < lowest_loss:
        lowest_loss = val_loss
        save_state(f"checkpoints/{args.project}.pt", model, optimizer, epoch, lowest_loss)

if val_loss < lowest_loss:
    lowest_loss = val_loss
    save_state(f"checkpoints/{args.project}.pt", model, optimizer, epoch, lowest_loss)

# save_state(f"checkpoints/{args.project}.pt",model, optimizer, epoch, lowest_loss)
