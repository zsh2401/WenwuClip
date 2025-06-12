#!/usr/bin/env bash
 python train.py --data-scale=0.01 --project=l16 --batch-size=32 --freeze-mode=b --base=ViT-B-16 --workers=0