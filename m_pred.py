import argparse
from pathlib import Path

import torch
from cn_clip.clip import load_from_name, tokenize
from PIL import Image

from application import classifier, probable
from dataset import load_data
from train_helpers import read_state

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, required=False)
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--base", type=str, required=True)
parser.add_argument("--device", type=str, required=False, default=None)
args = parser.parse_args()

if args.device is not None:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model, preprocess = load_from_name(args.base, device=device, download_root='./base')

model = model.to(device)

if args.project is not None:
    checkpoint = Path("checkpoints") / (args.project + ".pt")
    if checkpoint.exists():
        print("Loading from checkpoint", checkpoint)
        model_state = read_state(checkpoint, device)[0]
        model.load_state_dict(model_state)

if __name__ == "__main__":
    model.eval()
    category_prompts = ["是" + cate for cate in load_data()["categories"]]
    types_prompts = ["是" + cate for cate in load_data()["types"]]
    with torch.no_grad():
        images = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

        cate_prompt_tokens = tokenize(category_prompts).to(device, dtype=torch.long)
        types_prompt_tokens = tokenize(category_prompts).to(device, dtype=torch.long)

        image_feats = model.encode_image(images)
        cate_text_feats = model.encode_text(cate_prompt_tokens)
        types_text_feats = model.encode_text(types_prompt_tokens)

        logit_scale = model.logit_scale.clamp(max=4.6052)  # 必须狠狠冷静一下！

        classifier(category_prompts, image_feats, logit_scale, cate_text_feats)

        d,s = probable(types_prompts, image_feats, types_text_feats, logit_scale)
        print(s)
