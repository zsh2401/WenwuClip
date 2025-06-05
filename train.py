import argparse
from functools import cache
import os
from dataset import WenwuDataset
from torch.utils.data import DataLoader
from cn_clip.clip import load_from_name, tokenize
import torch
from torch import LongTensor, nn
import tqdm

from datetime import datetime
from train_helpers import freeze, train, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=6)
parser.add_argument("-l", "--lr", type=float, default=5e-7)
parser.add_argument("-c", "--checkpoint", type=str, default=None)
parser.add_argument("-d", "--device", type=str, default=None)
parser.add_argument("--base", type=str, default="ViT-H-14")
parser.add_argument("-s","--scale", type=float, default=1)
args = parser.parse_args()

if args.device is not None:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

train_dataset = WenwuDataset(0, 0 + (0.8 * args.scale))
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)

val_dataset = WenwuDataset(0.8, 0.8 + (0.1 * args.scale))
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)

criterion_img = nn.CrossEntropyLoss().to(device)
criterion_text = nn.CrossEntropyLoss().to(device)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = checkpoint["model"]
    optimizer = checkpoint["optimizer"]
    start_epoch = checkpoint["start_epoch"]
else:
    model, preprocess = load_from_name(args.base, device=device, download_root='./base')
    freeze(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    start_epoch = 1

for epoch in range(start_epoch, args.epochs + 1):
    train(f"{epoch}/{args.epochs}", model, train_loader, optimizer, device)
    accuracy, val_loss = evaluate(model, val_loader, device)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model": model,
        "optimizer": optimizer,
        "start_epoch": epoch
    }, f"checkpoints/{epoch}-{accuracy:.2%}-{val_loss:.4}-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.pt")
