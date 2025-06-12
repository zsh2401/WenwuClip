from typing import List, Dict, Tuple
import torch
from cn_clip.clip import tokenize
from cn_clip.clip.model import CLIP
from torch import Tensor
import tqdm


def encode_text_in_batch(model: CLIP, prompts: list[str], device: str, batch_size: int = 384) -> torch.Tensor:
    model.eval()
    feats_list: list[Tensor] = []  # 收集每批特征

    rng = range(0, len(prompts), batch_size)
    pbar = tqdm.tqdm(rng, desc="encode_text", unit="batch")

    for start in pbar:
        end = min(start + batch_size, len(prompts))
        sub_prompts = prompts[start:end]

        # --- Tokenize 在 CPU ---
        tokens = tokenize(sub_prompts).to(device)

        feats_list.extend(model.encode_text(tokens))

    return torch.stack(feats_list).to(device)


def print_classifier(prompts: list[str], probs, selected):
    # 4. 打印结果
    strs = []
    for i, prob in enumerate(probs):
        strs.append(f"{prompts[i]:<10} : {prob.item():5.2%}")

    print("\n".join(strs))
    print("\n预测结果:", prompts[selected])


def print_probable(prompts: list[str], probs, selected):
    strs = []
    for i, prob in enumerate(probs):
        strs.append(f"{prompts[i]:<10} : {prob:5.2%}")

    print("\nProb:", "\n".join(strs))
    print("\nSelected:", ', '.join([prompts[sel] for sel in selected]))


def classifier(prompts: list[str],
               image_feats,
               logit_scale,
               text_feats):
    # -- 全部转 float32，防止半精度溢出 --------------------------------------
    image_feats = image_feats.float()
    text_feats = text_feats.float()
    logit_scale = logit_scale.float()

    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)  # (1, D)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # (4, D)

    # 2. 计算对数相似度（CLIP InfoNCE 路径）
    logits_per_image = (logit_scale.exp() * image_feats @ text_feats.T)  # (1, 4)

    # 3. softmax 得到概率
    probs = logits_per_image.softmax(dim=-1).squeeze(0)  # (4,)
    pred_idx = probs.argmax().item()

    return probs, pred_idx


def probable(
        prompts: list[str],
        image_feats: torch.Tensor,  # (1, D)
        text_feats: torch.Tensor,  # (N_attr, D)
        logit_scale: torch.Tensor,  # scalar tensor
        top_k: int | None = 3,
        score_threshold: float | None = None,
):
    """
    多标签推断：返回 {属性: 置信度} + 满足过滤条件的属性列表
    参数
    ----
    top_k            : 仅取余弦得分最高的前 k 个；None 表示不限
    score_threshold  : 置信度阈值 (0-1)。二选一或都设也行
    """
    # 1. 归一化 + 求相似度
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    TEMP = 1
    scores = (TEMP * image_feats @ text_feats.T).squeeze(0)  # (N_attr,)

    # 2. Sigmoid → [0,1] 概率（互不排斥）
    probs = scores.sigmoid()
    print(probs)

    # 3. 组装字典
    # prob_dict = zip(prompts, probs)

    # 4. 过滤：Top-K & 阈值，两种逻辑可并存
    keep_mask = torch.ones_like(probs, dtype=torch.bool)

    if top_k is not None:
        top_idx = probs.topk(top_k).indices
        mask_topk = torch.zeros_like(probs, dtype=torch.bool)
        mask_topk[top_idx] = True
        keep_mask &= mask_topk

    if score_threshold is not None:
        keep_mask &= (probs >= score_threshold)

    selected_idx = [i for i, keep in enumerate(keep_mask) if keep]

    return probs, selected_idx
