import argparse
import math

import transformers
from dataset import WenwuDataset
from torch.utils.data import DataLoader
# from cn_clip import 
from cn_clip.clip.model import convert_models_to_fp32,convert_weights
from cn_clip.clip import load_from_name, available_models,tokenize
from cn_clip.clip.model import CLIP
import torch
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-b","--batch-size", type=int,default=32)
parser.add_argument("-e","--epochs", type=int,default=6)
parser.add_argument("-l","--lr",type=float,default=5e-12)
parser.add_argument("-d","--device",type=str,default=None)
parser.add_argument("-b","--base",type=str,default="ViT-H-14")
args = parser.parse_args()

if args.device is not None:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
train_dataset = WenwuDataset(0.5,0.5002)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,pin_memory=True)
model, preprocess = load_from_name(args.base, device=device, download_root='./')

def freeze(model:CLIP):
    for params in model.parameters():
        params.requires_grad = False
    # 冻结ViT-H/14前30层，解冻后几层
    for layer in model.visual.transformer.resblocks[30:]: 
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
    
freeze(model)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
criterion_img = nn.CrossEntropyLoss().to(device)
criterion_text = nn.CrossEntropyLoss().to(device)


for epoch in tqdm.tqdm(range(1, args.epochs + 1),position=0,desc="Epochs"):
    model.train()
    for images,texts in tqdm.tqdm(train_loader,position=1,desc="Training"):
        optimizer.zero_grad()
        
        images = images.to(device)
        tokenized_texts = tokenize(texts).to(device)
        
        image_feats, text_feats,logit_scale = model(images,tokenized_texts)
        
        # # 2. 直接点积再乘以温度
        logits_per_image = logit_scale * image_feats @ text_feats.t()  # [B, B]
        logits_per_text  = logits_per_image.t()                        # [B, B]
    
        # # 3. 交叉熵
        labels = torch.arange(images.size(0), device=device)
        loss_i2t = criterion_img(logits_per_image, labels)
        loss_t2i = criterion_text(logits_per_text,  labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
    print(f"Epoch {epoch} 完成，最后一个 batch 的 loss = {loss.item():.4f}")