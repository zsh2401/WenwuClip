import json
import os
import time
import types
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
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


import torch
from torch.utils.data import DataLoader
from typing import Sequence, Dict


@torch.no_grad()
def evaluate_clip_multicap(
        model,
        val_loader: DataLoader,  # 每 batch = (image, tokens, img_id)
        device: str | torch.device = "cuda",
        topk: Sequence[int] = (1, 5, 10),
        l2_norm: bool = True,
):
    """
    一图多文场景下计算 I→T / T→I Recall@K
    model      : CLIP / OpenCLIP / CN-CLIP，需有 encode_image / encode_text
    val_loader : 不 shuffle；Dataset 已将“一张图 × 多 caption” 展开
    """

    model.eval()

    # 1️⃣  把所有图片 / 文本向量提前算好
    img_feat_dict, img_id_order = {}, []  # id → embedding；保持顺序
    txt_feats, txt2img = [], []  # 文本 embedding；其归属 img_id

    for imgs, txt_tok, img_ids in tqdm.tqdm(val_loader, desc="Extracting features."):
        imgs, txt_tok, img_ids = imgs.to(device), txt_tok.to(device), img_ids.to(device)

        f_img = model.encode_image(imgs)
        f_txt = model.encode_text(txt_tok)

        if l2_norm:
            f_img = f_img / f_img.norm(dim=-1, keepdim=True)
            f_txt = f_txt / f_txt.norm(dim=-1, keepdim=True)

        # 同一张图只保留一次特征
        for j, gid in enumerate(img_ids):
            g = int(gid)
            if g not in img_feat_dict:
                img_feat_dict[g] = f_img[j]
                img_id_order.append(g)

        txt_feats.append(f_txt)
        txt2img.extend(img_ids.tolist())

    image_embs = torch.stack([img_feat_dict[i] for i in img_id_order]).to(device)  # (N_img, D)
    text_embs = torch.cat(txt_feats, dim=0)  # (N_txt, D)
    txt2img_t = torch.tensor(txt2img, device=device)  # (N_txt,)

    # 2️⃣  建立【图片 ID ↔ 行号】互映
    id2row = {img_id: row for row, img_id in enumerate(img_id_order)}  # id → 行号
    row2id_t = torch.tensor(img_id_order, device=device)  # 行号 → id
    txt2row = torch.tensor([id2row[i] for i in txt2img], device=device)  # 文本 → 行号

    # 3️⃣  相似度矩阵
    sim_i2t = image_embs @ text_embs.T  # (N_img, N_txt)
    sim_t2i = sim_i2t.T  # (N_txt, N_img)

    results = {
        "i2t": [],
        "t2i": [],
        "score": 0,
    }
    for k in topk:
        try:
            # —— I → T（按行求 top-k caption）
            idx_top_txt = sim_i2t.topk(k, dim=-1).indices  # (N_img, k)
            hit_i2t = (txt2row[idx_top_txt] == torch.arange(
                sim_i2t.size(0), device=device).unsqueeze(1)).any(dim=1)
            results["i2t"].append(hit_i2t.float().mean().item())

            # —— T → I（按行求 top-k image）
            idx_top_img = sim_t2i.topk(k, dim=-1).indices  # (N_txt, k)
            hit_t2i = (idx_top_img == txt2row.unsqueeze(1)).any(dim=1)
            results["t2i"].append(hit_t2i.float().mean().item())

        except Exception as e:
            print(e)

    scores = results["t2i"] + results["t2i"]
    results["score"] = sum(scores) / len(scores)
    return results


def evaluate(model: CLIP,
             val_loader: DataLoader,
             device: DeviceObjType,
             criterion_img: Module = CrossEntropyLoss(),
             criterion_text: Module = CrossEntropyLoss(),
             ):
    model.eval()
    with tqdm.tqdm(total=len(val_loader), desc=f"Validating") as val_bar:
        val_bar.set_postfix({
            "acc": f"00%",
            "val_loss": f"0.0000"
        })
        # 验证：
        correct, total = 0, 0
        with torch.no_grad():
            for images, text_tokens, img_ids in val_loader:
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
                    "loss": f"{val_loss:.4f}"
                })
                val_bar.update(1)

        return accuracy, val_loss


def get_loss(model: CLIP, images, text_tokens,
             criterion_img: Module = CrossEntropyLoss(),
             criterion_text: Module = CrossEntropyLoss()):
    image_feats, text_feats, logit_scale = model(images, text_tokens)

    # F
    # # 2. 直接点积再乘以温度
    logits_per_image = logit_scale * image_feats @ text_feats.t()  # [B, B]
    logits_per_text = logits_per_image.t()  # [B, B]

    # # 3. 交叉熵
    labels = torch.arange(images.size(0), device=images.device)
    loss_i2t = criterion_img(logits_per_image, labels)
    loss_t2i = criterion_text(logits_per_text, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss

def inspect_model_dtype(model):
    import collections, torch
    cnt = collections.Counter(p.dtype for p in model.parameters())
    print("### Parameter dtypes:")
    for dtype, n in cnt.items():
        print(f"  {dtype}: {n} tensors")

    # 看首个可训练参数
    first = next(p for p in model.parameters() if p.requires_grad)
    print("\nfirst trainable param dtype :", first.dtype)

    # 如果已经做过一次反向，还能看梯度 dtype
    if first.grad is not None:
        print("first grad dtype             :", first.grad.dtype)

def move_model_to(module: CLIP, device, precision):
    if precision == "fp32":
        return module.to(device, dtype=torch.float32)
    elif precision == "fp16":
        return module.to(device, dtype=torch.float16)
    else:
        # 确保模型完全处于fp32精度，否则将无法进行自动精度转换的训练。
        # 这个非常重要，如果不这么做，模型中有部分tensor的精度是float16，
        # 这会导致后续的计算要么出现nan，要么Scaler无法工作。
        return module.to(device).float()


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
    losses = []
    with tqdm.tqdm(total=len(train_loader), desc=bar_prefix) as epoch_bar:
        epoch_bar.set_postfix(loss=f"0.0000")
        for images, text_tokens, img_ids in train_loader:
            optimizer.zero_grad()

            images = images.to(device)
            text_tokens = text_tokens.to(device)

            if use_amp:

                # 使用半精度进行forward
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = get_loss(model, images, text_tokens, criterion_img, criterion_text)

                # 进行缩放
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = get_loss(model, images, text_tokens, criterion_img, criterion_text)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            epoch_bar.update(1)
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

    # 计算平均loss
    return sum(losses) / len(losses)


def read_reports(projectname: str):
    try:
        c = Path("checkpoints")
        c.mkdir(exist_ok=True)
        with open(f"{c / projectname}.reports.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "history": []
        }


def save_reports(projname: str, reports):
    c = Path("checkpoints")
    c.mkdir(exist_ok=True)
    with open(f"{c / projname}.reports.json", "w") as f:
        json.dump(reports, f, indent=4, ensure_ascii=False)


def read_state(filename: str, device: str):
    checkpoint = torch.load(filename, map_location=device)
    model_state = checkpoint["model"]
    optimizer_state = checkpoint["optimizer"]
    start_epoch = checkpoint["start_epoch"]
    scaler = checkpoint["scaler"]
    return model_state, optimizer_state, scaler, start_epoch


def save_state(filename: str,
               model: CLIP,
               optimizer: Optimizer,
               epoch: int,
               scaler: GradScaler
               ):
    print("Saving...")
    start = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "start_epoch": epoch,
        "scaler": scaler.state_dict(),
    }, filename)
    end = time.time()
    print(f"Saved checkpoint at epoch {epoch} at {end - start} seconds")
