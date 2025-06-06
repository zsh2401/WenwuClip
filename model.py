from cn_clip.clip import load_from_name
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn import CrossEntropyLoss

from train_helpers import get_loss


class WenWuClip(pl.LightningModule):
    def __init__(self, base: str, freeze_strategy: str):
        super().__init__()
        model, preprocess = load_from_name(base, download_root='./base')
        self.clip = model
        self.criterion_img = CrossEntropyLoss()
        self.criterion_text = CrossEntropyLoss()

    def forward(self, imgs, text_tokens):
        return self.clip(imgs, text_tokens)

    def training_step(self, batch, batch_idx):
        images, text_tokens, img_ids = batch
        return get_loss(self.clip, images, text_tokens)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return
