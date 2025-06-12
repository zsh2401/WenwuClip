from cn_clip.clip.model import CLIP


def freeze(model: CLIP, strategy: str):
    if strategy == "a":
        freeze_a(model)
    elif strategy == "b":
        freeze_b(model)
    elif strategy == "all":
        print("Freezing all layers")
        freeze_all(model)
    else:
        print("Unfreeze all layers")

    # 3. 简单检查：打印出哪些模块是可训练的
    trainable_layers = []
    frozen_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
            # print(f"\tTrainable\t{name}")
        else:
            frozen_layers.append(name)
            # print(f"\tFreezed..\t{name}")

    print("Trainable layers: {}".format(len(trainable_layers)))
    print("Frozen layers: {}".format(len(frozen_layers)))

    # 4. 建议：计算一下可训练参数占比
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters / Total = {trainable_params} / {total_params} = {trainable_params / total_params:.2%}")


def freeze_all(model: CLIP):
    for params in model.parameters():
        params.requires_grad = False


def freeze_b(model: CLIP):
    # return
    for params in model.parameters():
        params.requires_grad = False

    # 冻结ViT-H/14前30层，解冻后几层
    for layer in model.visual.transformer.resblocks[-8:]:
        for param in layer.parameters():
            param.requires_grad = True

    for layer in model.bert.encoder.layer[-8:]:
        for param in layer.parameters():
            param.requires_grad = True

    # 确保logit_scale解冻（如需冻结则设False）
    model.logit_scale.requires_grad = True


def freeze_a(model: CLIP):
    # return
    for params in model.parameters():
        params.requires_grad = False

    # 冻结ViT-H/14前30层，解冻后几层
    for layer in model.visual.transformer.resblocks[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    for layer in model.bert.encoder.layer[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # 确保logit_scale解冻（如需冻结则设False）
    model.logit_scale.requires_grad = True
