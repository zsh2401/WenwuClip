import json
import time
import types
from typing import Dict, Sequence

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


@torch.no_grad()
def evaluate_clip_multicap(
        model: CLIP,  # ✦ CLIP / CN-CLIP / OpenCLIP 模型
        data_loader: DataLoader,  # ✦ batch = (img_tensor, text_tokens_tensor, img_id)
        device: str | torch.device,
        topk: Sequence[int] = (1, 5, 10),
        l2_norm: bool = True,
) -> Dict[str, float]:
    """
    计算多文一图场景下的 I→T / T→I Recall@K

    -------------
    数据假设
    -------------
      • (image, caption, img_id) 已在 Dataset 中“一张图 × N 条 caption” 全展开
      • 同一张图在不同样本里 img_id 相同

    -------------
    返回值示例
    -------------
      {'i2t_R1': 11.23, 'i2t_R5': 28.74, 'i2t_R10': 39.85,
       't2i_R1': 10.17, 't2i_R5': 26.02, 't2i_R10': 36.44}
    """

    model.eval()

    # ---------- Step 1: 前向提特征 ----------
    img_feat_dict: dict[int, torch.Tensor] = {}  # img_id ➜ feature
    img_id_order: list[int] = []  # 记录唯一图片的顺序
    txt_feats = []  # 全部文本特征
    txt2img = []  # 每条文本对应的 img_id

    for imgs, txt_tokens, img_ids in tqdm.tqdm(data_loader, desc="Inferencing input embeddings"):
        imgs, txt_tokens, img_ids = (imgs.to(device),
                                     txt_tokens.to(device),
                                     img_ids.to(device))

        f_img = model.encode_image(imgs)
        f_txt = model.encode_text(txt_tokens)

        if l2_norm:
            f_img = f_img / f_img.norm(dim=-1, keepdim=True)
            f_txt = f_txt / f_txt.norm(dim=-1, keepdim=True)

        # 收集唯一图像特征
        for j, gid in enumerate(img_ids):
            gid_int = int(gid.item())
            if gid_int not in img_feat_dict:
                img_feat_dict[gid_int] = f_img[j]
                img_id_order.append(gid_int)

        txt_feats.append(f_txt)
        txt2img.extend(img_ids.tolist())

    image_embs = torch.stack([img_feat_dict[i] for i in img_id_order]).to(device)  # (N_img, D)
    text_embs = torch.cat(txt_feats, dim=0)  # (N_txt, D)
    txt2img_t = torch.tensor(txt2img, device=device)  # (N_txt,)
    row_img_id = torch.tensor(img_id_order, device=device)  # (N_img,)

    # ---------- Step 2: 相似度矩阵 ----------
    sim_i2t = image_embs @ text_embs.T  # (N_img, N_txt)
    sim_t2i = sim_i2t.T  # (N_txt, N_img)

    # ---------- Step 3: 计算 Recall ----------
    results: Dict[str, float] = {}
    for k in topk:
        # I → T ：检查任意 top-k caption 的 img_id 是否等于查询图像
        idx_top_txt = sim_i2t.topk(k, dim=-1).indices  # (N_img, k)
        hit_i2t = (txt2img_t[idx_top_txt] == row_img_id.unsqueeze(1)).any(dim=1)
        results[f"i2t_R{k}"] = hit_i2t.float().mean().item() * 100

        # T → I ：查询 caption 的正确图像在 top-k 内？
        idx_top_img = sim_t2i.topk(k, dim=-1).indices  # (N_txt, k)
        hit_t2i = (idx_top_img == txt2img_t.unsqueeze(1)).any(dim=1)
        results[f"t2i_R{k}"] = hit_t2i.float().mean().item() * 100

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

    # # 2. 直接点积再乘以温度
    logits_per_image = logit_scale * image_feats @ text_feats.t()  # [B, B]
    logits_per_text = logits_per_image.t()  # [B, B]

    # # 3. 交叉熵
    labels = torch.arange(images.size(0), device=images.device)
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
        for images, text_tokens, img_ids in train_loader:
            optimizer.zero_grad()

            images = move_to(images, device, precision)
            text_tokens = move_to(text_tokens, device, precision)

            if use_amp:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    # print(f"{device} float 16")
                    loss = get_loss(model, images, text_tokens, criterion_img, criterion_text)
                    # print(loss.dtype)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = get_loss(model, images, text_tokens, criterion_img, criterion_text)
                loss.backward()
                optimizer.step()

            epoch_bar.update(1)
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")


def read_reports(projectname: str):
    with open(f"{projectname}.reports.json", "r") as f:
        return json.load(f)


def save_reports(projname: str, reports):
    with open(f"{projname}.reports.json", "w") as f:
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
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "start_epoch": epoch,
        "scaler": scaler.state_dict(),
        "epoch": epoch
    }, filename)
    end = time.time()
    print(f"Saved checkpoint at epoch {epoch} at {end - start} seconds")
