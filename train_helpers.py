import json
import time
import types

import torch
from cn_clip.clip.model import CLIP
import tqdm
from torch import DeviceObjType, GradScaler
from torch.nn import Module, CrossEntropyLoss

from torch.utils.data import DataLoader
from torch.optim import Optimizer


def tackle(model: CLIP):
    def fp32_mha_forward(self, q, k, v, **kw):
        # 临时把 Q、K、V cast 成 fp32，算完再 cast 回原 dtype
        out_dtype = q.dtype
        q = q.float();
        k = k.float();
        v = v.float()
        attn_out, _ = self._orig_forward(q, k, v, **kw)  # ← 这里做的是 scaled_dot_product_attention
        return attn_out.to(out_dtype)

    for mod in model.modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod._orig_forward = mod.forward  # 备份原 forward
            mod.forward = types.MethodType(fp32_mha_forward, mod)

    # def hooks(module, inp, out):
    #     if isinstance(out, tuple): out = out[0]
    #     if torch.isnan(out).any() or torch.isinf(out).any():
    #         raise RuntimeError(f"NAN at {module.__class__.__name__}")
    #
    # for m in model.modules():
    #     m.register_forward_hook(hooks)

    # for m in model.modules():  # LN & Softmax 强制 FP32
    #     if isinstance(m, (torch.nn.LayerNorm, torch.nn.Softmax)):
    #         m.float()


def freeze(model: CLIP):
    # return
    for params in model.parameters():
        params.requires_grad = False
    # 冻结ViT-H/14前30层，解冻后几层
    for layer in model.visual.transformer.resblocks[28:]:
        for param in layer.parameters():
            param.requires_grad = True

    for layer in model.bert.encoder.layer[20:]:
        for param in layer.parameters():
            param.requires_grad = True

    # 确保logit_scale解冻（如需冻结则设False）
    model.logit_scale.requires_grad = True

    # 3. 简单检查：打印出哪些模块是可训练的
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\tTrainable\t{name}")
        else:
            print(f"\tFreezed..\t{name}")

    # 4. 建议：计算一下可训练参数占比
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters / Total = {trainable_params} / {total_params} = {trainable_params / total_params:.2%}")


def evaluate(model: CLIP,
             val_loader: DataLoader,
             device: DeviceObjType,
             criterion_img: Module = CrossEntropyLoss(),
             criterion_text: Module = CrossEntropyLoss(),
             ):
    model.eval()
    with tqdm.tqdm(total=len(val_loader), desc=f"Validating") as val_bar:
        # 验证：
        correct, total = 0, 0
        with torch.no_grad():
            for images, text_tokens in val_loader:
                val_bar.set_postfix({
                    "acc": f"00%",
                    "val_loss": f"0.0000"
                })
                images = images.to(device)
                text_tokens = text_tokens.to(device)
                image_feats, text_feats, logit_scale = model(images, text_tokens)

                # # 2. 直接点积再乘以温度
                logits_per_image = logit_scale * image_feats @ text_feats.t()  # [B, B]
                logits_per_text = logits_per_image.t()  # [B, B]

                preds = logits_per_image.argmax(dim=1)
                labels = torch.arange(logits_per_image.size(0), device=device)

                loss_i2t = criterion_img(logits_per_image, labels)
                loss_t2i = criterion_text(logits_per_text, labels)
                val_loss = (loss_i2t + loss_t2i) / 2

                correct += (preds == labels).sum().item()
                total += logits_per_image.size(0)
                accuracy = correct / total
                val_bar.set_postfix({
                    "acc": f"{accuracy:.2%}",
                    "val_loss": f"{val_loss:.4f}"
                })
                val_bar.update(1)

        return accuracy, val_loss


def get_loss(model: CLIP, images, text_tokens, device,
             criterion_img: Module = CrossEntropyLoss(),
             criterion_text: Module = CrossEntropyLoss(), ):
    image_feats, text_feats, logit_scale = model(images, text_tokens)

    # # 2. 直接点积再乘以温度
    logits_per_image = logit_scale * image_feats @ text_feats.t()  # [B, B]
    logits_per_text = logits_per_image.t()  # [B, B]

    # # 3. 交叉熵
    labels = torch.arange(images.size(0), device=device)
    loss_i2t = criterion_img(logits_per_image, labels)
    loss_t2i = criterion_text(logits_per_text, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss


def move_to(module: Module, device, precision):
    if precision == "fp32":
        return module.to(device, dtype=torch.float32)
    elif precision == "fp16":
        return module.to(device, dtype=torch.float16)
    else:
        return module.to(device)


def train(bar_prefix: str,
          model: CLIP, train_loader: DataLoader, optimizer: Optimizer,
          device: DeviceObjType,
          use_amp: bool,
          precision: str,
          scaler: GradScaler,
          criterion_img: Module = CrossEntropyLoss(),
          criterion_text: Module = CrossEntropyLoss(),
          ):
    model.train()

    with tqdm.tqdm(total=len(train_loader), desc=bar_prefix) as epoch_bar:
        epoch_bar.set_postfix(loss=f"0.0000")
        for images, text_tokens in train_loader:
            optimizer.zero_grad()

            images = move_to(images, device, precision)
            text_tokens = move_to(text_tokens, device, precision)

            if use_amp:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    # print(f"{device} float 16")
                    loss = get_loss(model, images, text_tokens, device, criterion_img, criterion_text)
                    # print(loss.dtype)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = get_loss(model, images, text_tokens, device, criterion_img, criterion_text)
                loss.backward()
                optimizer.step()

            epoch_bar.update(1)
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")


def read_reports(projectname: str):
    with open(f"{projectname}.reports.json", "r") as f:
        return json.load(f)


def save_reports(projname: str, reports):
    with open(f"{projname}.reports.json", "w") as f:
        json.dump(reports, f)


def read_state(filename: str, device: DeviceObjType):
    checkpoint = torch.load(filename, map_location=device)
    model = checkpoint["model"]
    optimizer = checkpoint["optimizer"]
    start_epoch = checkpoint["start_epoch"]
    lowest_loss = checkpoint["lowest_loss"]
    return model, optimizer, start_epoch, lowest_loss


def save_state(filename: str,
               model: CLIP,
               optimizer: Optimizer,
               epoch: int,
               lowest_loss: float,
               ):
    print("Saving...")
    start = time.time()
    torch.save({
        "model": model,
        "optimizer": optimizer,
        "start_epoch": epoch,
        "lowest_loss": lowest_loss,
    }, filename)
    end = time.time()
    print(f"Saved checkpoint at epoch {epoch} and lowest loss {lowest_loss} at {end - start} seconds")
