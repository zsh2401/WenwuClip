import pytorch_lightning as pl
import torch
from cn_clip.clip import load_from_name
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import OptimizerLRScheduler, TRAIN_DATALOADERS
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from freeze_strategies import freeze
from train_helpers import get_loss
import torch.functional as F


class WenWuClip(pl.LightningModule):
    def __init__(self, base: str, lr: float):
        super().__init__()
        model, preprocess = load_from_name(base, download_root='./base')
        self.clip = model
        self.lr = lr
        self.val_losses = []

    def my_freeze(self, freeze_strategy: str):
        freeze(self.clip, freeze_strategy)

    def forward(self, imgs, text_tokens):
        return self.clip(imgs, text_tokens)

    def on_epoch_start(self):
        print('\n')

    def training_step(self, batch, batch_idx):
        images, text_tokens, img_ids = batch
        loss = get_loss(self.clip, images, text_tokens)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, text_tokens, img_ids = batch
        loss = get_loss(self.clip, images, text_tokens)
        self.val_losses.append(loss.item())
        return loss

    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        self.log("Wonderful", avg_loss)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        trainable_params = [p for p in self.clip.parameters() if p.requires_grad]
        return AdamW(trainable_params, lr=self.lr, weight_decay=0.01)
