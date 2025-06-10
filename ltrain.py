import argparse

import pytorch_lightning
import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from torch.utils.data import DataLoader

from dataset import WenwuDataset
from model import WenWuClip

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=6)
parser.add_argument("-l", "--lr", type=float, default=5e-7)
parser.add_argument("-c", "--checkpoint", type=str, default=None)
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
train_loader = DataLoader(train_dataset,
                          persistent_workers=True,
                          num_workers=args.workers, shuffle=True, batch_size=args.batch_size,
                          pin_memory=True)

val_dataset = WenwuDataset(0.8, 0.8 + (0.1 * args.data_scale))
val_loader = DataLoader(val_dataset, num_workers=args.workers,
                        persistent_workers=True,
                        shuffle=True, batch_size=args.batch_size,
                        pin_memory=True)

if __name__ == "__main__":
    pytorch_lightning.seed_everything(2401)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,  # 保存最后一个 epoch
        dirpath="checkpoints/",  # 保存目录
        mode="min",
        filename=args.project,  # 文件名 = last.ckpt
    )
    trainer = Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback,TQDMProgressBar(leave=True)])
    if args.checkpoint is not None:
        model = WenWuClip.load_from_checkpoint(args.checkpoint)
        model.my_freeze(args.freeze_mode)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
    else:
        model = WenWuClip(args.base, args.lr)
        model.my_freeze(args.freeze_mode)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
