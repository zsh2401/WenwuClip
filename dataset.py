from functools import cache
from pathlib import Path
import json

import torch
from PIL import Image
import math
import tqdm
from torchvision import transforms

# CLIP 官方预训练时常用的归一化参数（RGB 三个通道的均值和标准差）
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

@cache
def load_data():
    data_json_path:str="./dataset/data.json.txt"
    images_root = Path("./dataset/")
    with open(data_json_path,'r') as f:
        items = json.load(f)["items"]
        ids = sorted(items.keys())
    final_data = []
    for id in tqdm.tqdm(ids,desc="Building captions"):
        for caption in build_captions(items[id]):
            if "img_paths" not in items[id]:
                print(f"No images found for {items[id]["name"]}")
            else:
                for img_path in items[id]["img_paths"]:
                    final_data.append((
                        id,images_root / img_path, caption
                    ))
        
    return final_data

def build_captions(item)->list[str]:
    result:list[str] = []
    result.append( f"{item.get("name")},{item["meta"]["年代"]},{item["meta"]["分类"]}")
    if item["patterns"]:
        result.append( f"{",".join(item["patterns"])}")
    if item["types"]:
        result.append( f"{",".join(item["types"])}")
    return result

class WenwuDataset:
    def __init__(self,start_p:float,end_p:float):
        self.data = load_data()
        self.data = self.data[math.floor(start_p * len(self.data)):math.floor(end_p * len(self.data))]
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(3/4,4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN,CLIP_STD)
        ])
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, image, caption =  self.data[idx]
        image = Image.open(image).convert('RGB')
        image_tensor = self.transform(image)
        assert not torch.isnan(image_tensor).any(), "Image tensor contains NaN"
        assert not torch.isinf(image_tensor).any(), "Image tensor contains Inf"
        assert not "" == caption.strip(), "Caption is empty"
        return image_tensor, caption