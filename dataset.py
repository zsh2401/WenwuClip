from functools import cache
import os
from pathlib import Path
import json

import torch
from PIL import Image
import math
import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

from cn_clip.clip import tokenize

# CLIP 官方预训练时常用的归一化参数（RGB 三个通道的均值和标准差）
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD)
])


@cache
def load_data():
    data_json_path: str = "./dataset/data.json.txt"
    images_root = Path("./dataset/")
    with open(data_json_path, 'r') as f:
        items = json.load(f)["items"]
        ids = sorted(items.keys())
    final_data = []
    next_id = 0
    path2id: dict[str, int] = {}
    for id in tqdm.tqdm(ids, desc="Building captions"):
        for caption in build_captions(items[id]):
            if "img_paths" not in items[id]:
                print(f"No images found for {items[id]["name"]}")
            else:
                for img_path in items[id]["img_paths"]:
                    if img_path not in path2id:
                        path2id[img_path] = next_id
                        next_id += 1
                    img_id = path2id[img_path]
                    dest = images_root / img_path
                    final_data.append((
                        id, dest, caption, img_id
                    ))

    return final_data


def build_captions(item) -> list[str]:
    result: list[str] = []
    result.append(f"{item.get("name")},{item["meta"]["年代"]},{item["meta"]["分类"]}")
    if item["patterns"]:
        result.append(f"{",".join(item["patterns"])}")
    if item["types"]:
        result.append(f"{",".join(item["types"])}")
    return result


@cache
def get_image(path: str):
    return Image.open(path).convert('RGB')


@cache
def cached_tokenized(text):
    return tokenize([text]).squeeze(0)


class WenwuDataset(Dataset):
    def __init__(self, start_p: float, end_p: float, img_in_memory=False):
        super().__init__()
        self.data = load_data()
        self.img_in_memory = img_in_memory
        self.data = self.data[math.floor(start_p * len(self.data)):math.floor(end_p * len(self.data))]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, image, caption, img_id = self.data[idx]
        if self.img_in_memory:
            image = get_image(str(image))
        else:
            image = Image.open(image).convert('RGB')
        image_tensor = self.transform(image)
        assert not torch.isnan(image_tensor).any(), "Image tensor contains NaN"
        assert not torch.isinf(image_tensor).any(), "Image tensor contains Inf"
        assert not "" == caption.strip(), "Caption is empty"
        # tokens = tokenize([caption]).squeeze(0)
        return image_tensor, cached_tokenized(caption), img_id


if __name__ == "__main__":
    print("Building captions dataset...")
    print(WenwuDataset(0, 1)[10])
