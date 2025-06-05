from cn_clip.clip.model import CLIP


def freeze(model:CLIP):
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
    print(f"Trainable Parameters / Total = {trainable_params} / {total_params} = {trainable_params/total_params:.2%}")