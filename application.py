from typing import List, Dict, Tuple
import torch


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

    # 4. 打印结果
    for p, prob in zip(prompts, probs):
        print(f"{p:<10} : {prob.item() * 100:5.2f}%")

    pred_idx = probs.argmax().item()
    print("\n预测结果:", prompts[pred_idx])


def probable(
        prompts: list[str],
        image_feats: torch.Tensor,  # (1, D)
        text_feats: torch.Tensor,  # (N_attr, D)
        logit_scale: torch.Tensor,  # scalar tensor
        top_k: int | None = 3,
        score_threshold: float | None = None,
) -> Tuple[Dict[str, float], List[str]]:
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

    scores = (logit_scale.exp() * image_feats @ text_feats.T).squeeze(0)  # (N_attr,)

    # 2. Sigmoid → [0,1] 概率（互不排斥）
    probs = scores.sigmoid()

    # 3. 组装字典
    prob_dict = {p: float(prob) for p, prob in zip(prompts, probs)}

    # 4. 过滤：Top-K & 阈值，两种逻辑可并存
    keep_mask = torch.ones_like(probs, dtype=torch.bool)

    if top_k is not None:
        top_idx = probs.topk(top_k).indices
        mask_topk = torch.zeros_like(probs, dtype=torch.bool)
        mask_topk[top_idx] = True
        keep_mask &= mask_topk

    if score_threshold is not None:
        keep_mask &= (probs >= score_threshold)

    selected = [prompts[i] for i, keep in enumerate(keep_mask) if keep]

    return prob_dict, selected
