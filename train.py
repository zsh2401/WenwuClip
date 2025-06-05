import argparse
from functools import cache
import os
from dataset import WenwuDataset
from torch.utils.data import DataLoader
from cn_clip.clip import load_from_name, tokenize
import torch
from torch import LongTensor, nn
import tqdm

from datetime import datetime
from train_helpers import freeze

parser = argparse.ArgumentParser()
parser.add_argument("-b","--batch-size", type=int,default=32)
parser.add_argument("-e","--epochs", type=int,default=6)
parser.add_argument("-l","--lr",type=float,default=5e-7)
parser.add_argument("-c","--checkpoint",type=str,default=None)
parser.add_argument("-d","--device",type=str,default=None)
parser.add_argument("--base",type=str,default="ViT-H-14")
args = parser.parse_args()

if args.device is not None:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
train_dataset = WenwuDataset(0.3,0.4)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,pin_memory=True)

val_dataset = WenwuDataset(0.8,0.82)
val_loader = DataLoader(val_dataset,shuffle=True,batch_size=args.batch_size,pin_memory=True)


if args.checkpoint is not None:
    checkpoint =  torch.load(args.checkpoint,map_location=device)
    model = checkpoint["model"]
    optimizer = checkpoint["optimizer"]
    criterion_img = checkpoint["criterion_img"]
    criterion_text = checkpoint["criterion_text"]
    start_epoch = checkpoint["start_epoch"]
else:
    model, preprocess = load_from_name(args.base, device=device, download_root='./base')
    freeze(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion_img = nn.CrossEntropyLoss().to(device)
    criterion_text = nn.CrossEntropyLoss().to(device)
    start_epoch = 1

@cache
def cached_tokenize(texts:list[str])->LongTensor:
    return tokenize(texts)

for epoch in range(start_epoch, args.epochs + 1):
    model.train()
    
    with tqdm.tqdm(total=len(train_loader),desc=f"Epoch {epoch}/{args.epochs}") as epoch_bar:
        for images,text_tokens in train_loader:
            epoch_bar.set_postfix(loss=f"0.0000")
            optimizer.zero_grad()
        
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            
            image_feats, text_feats,logit_scale = model(images,text_tokens)
            
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
            
            epoch_bar.update(1)
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    # epoch_bar.close()
    
    with tqdm.tqdm(total=len(val_loader),desc=f"Validating") as val_bar:
        model.eval()
        # 验证：
        correct,total = 0,0
        with torch.no_grad():
            for images,text_tokens in val_loader:   
                val_bar.set_postfix({
                    "acc":f"00%",
                    "val_loss":f"0.0000"
                })
                images = images.to(device)
                tokenized_texts = tokenize(text_tokens).to(device)
                image_feats, text_feats, logit_scale = model(images,tokenized_texts)
                
                # # 2. 直接点积再乘以温度
                logits_per_image = logit_scale * image_feats @ text_feats.t()  # [B, B]
                logits_per_text  = logits_per_image.t()                        # [B, B]
                
                preds = logits_per_image.argmax(dim=1)
                labels = torch.arange(logits_per_image.size(0), device=device)
                
                loss_i2t = criterion_img(logits_per_image, labels)
                loss_t2i = criterion_text(logits_per_text,  labels)
                val_loss = (loss_i2t + loss_t2i) / 2
                
                correct += (preds == labels).sum().item()
                total += logits_per_image.size(0)
                accuracy = correct / total
                val_bar.set_postfix({
                    "acc":f"{accuracy:.2%}",
                    "val_loss":f"{val_loss:.4f}"
                })
                val_bar.update(1)
                
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model":model,
        "optimizer":optimizer,
        "criterion_img":criterion_img,
        "criterion_text": criterion_text,
        "start_epoch":epoch
        },f"checkpoints/{epoch}-{accuracy:.2%}-{val_loss:.4}-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.pt")