import argparse

import torch
from cn_clip.clip import load_from_name
from torch.utils.data import DataLoader

from dataset import WenwuDataset
from train_helpers import read_state, evaluate_clip_multicap, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-c", "--checkpoint", type=str, default=None)
parser.add_argument("-d", "--device", type=str, default=None)
parser.add_argument("--data-scale", type=float, default=1)
parser.add_argument("--base", type=str, default="ViT-H-14")
args = parser.parse_args()

if args.device is not None:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

test_dataset = WenwuDataset(0.9, 0.9 + (0.1 * args.data_scale))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

model, preprocess = load_from_name(args.base, device=device, download_root='./base')
if args.checkpoint is not None:
    model_state = read_state(args.checkpoint,device)[0]
    model.load_state_dict(model_state)
else:
    print("Using vanilla model for testing")
model.to(device)

# evaluate(model, test_loader, device)

scores = evaluate_clip_multicap(model, test_loader, device)
print("Image → Text  Recall:", {k:v for k,v in scores.items() if k.startswith('i2t')})
print("Text  → Image Recall:", {k:v for k,v in scores.items() if k.startswith('t2i')})
