from functools import cache
import os
from pathlib import Path
import json

import torch
from PIL import Image
import math
import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import v2 as TV

from cn_clip.clip import tokenize

# CLIP 官方预训练时常用的归一化参数（RGB 三个通道的均值和标准差）
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@cache
def get_index_data():
    data_json_path: str = "./dataset/data.json.txt"
    images_root = Path("./dataset/")
    with open(data_json_path, 'r') as f:
        items = json.load(f)["items"]
        ids = sorted(items.keys())
    dynasties = set()
    categories = set()
    types = set()
    final_data = []
    next_id = 0
    path2id: dict[str, int] = {}
    images = set()
    for id in tqdm.tqdm(ids, desc="Indexing dataset"):
        item = items[id]
        dynasties.add(item["meta"]["年代"])
        categories.add(item["meta"]["分类"])
        types.update(item["types"])
        for caption in build_captions(item):
            if "img_paths" not in item:
                print(f"No images found for {item["name"]}")
            else:
                for img_path in item["img_paths"]:
                    images.add(img_path)
                    if img_path not in path2id:
                        path2id[img_path] = next_id
                        next_id += 1
                    img_id = path2id[img_path]
                    dest = images_root / img_path
                    final_data.append((
                        id, dest, caption, img_id
                    ))

    return {
        "index": final_data,
        "dynasties": dynasties,
        "categories": categories,
        "types": types,
        "images" : images
    }


def heat():
    index_data = get_index_data()
    images = index_data["images"]
    images_root = Path("./dataset/")
    for image_path in tqdm.tqdm(images, desc="Heating images"):
        # id, image, caption, img_id = self.data[i]
        get_image(str(images_root / image_path))

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


def decorator_timer(some_function):
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time() - t1
        print(f"Time taken: {end}")
        return result

    return wrapper


class WenwuDataset(Dataset):
    def __init__(self, start_p: float,
                 end_p: float,
                 memorize_images=False,
                 img_preprocess=None,
                 device: str = "cpu"):
        super().__init__()
        self.img_preprocess = img_preprocess
        self.memorize_images = memorize_images
        indexing = get_index_data()["index"]
        self.data = indexing[math.floor(start_p * len(indexing)):math.floor(end_p * len(indexing))]
        self.transform = TV.Compose([
            TV.ToImage(),
            TV.ToDtype(torch.float32, scale=True),
            TV.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
            TV.RandomHorizontalFlip(),
            TV.Normalize(CLIP_MEAN, CLIP_STD),
        ]).to(device)

    def __len__(self):
        return len(self.data)

    # @decorator_timer
    def __getitem__(self, idx):
        id, image, caption, img_id = self.data[idx]
        if self.memorize_images:
            image = get_image(str(image))
        else:
            image = Image.open(image).convert('RGB')

        if self.img_preprocess:
            image_tensor = self.img_preprocess(image)
        else:
            image_tensor = self.transform(image)

        return image_tensor, cached_tokenized(caption), img_id


if __name__ == "__main__":
    print("Building captions dataset...")
    print(WenwuDataset(0, 1)[10])
