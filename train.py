#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import tqdm
from cn_clip.clip import load_from_name
from cn_clip.clip.model import CLIP
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

from dataset import WenwuDataset, heat
from dist_utils import setup_distributed
from utils import determine_device

torch.backends.verbose = True
torch.backends.cudnn.benchmark = True
from train_helpers import train, save_state, read_state, read_reports, save_reports, move_to, \
    evaluate_clip_multicap
from freeze_strategies import freeze


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-l", "--lr", type=float, default=5e-7)
    parser.add_argument("--memorize-images", type=bool, default=False)
    parser.add_argument("--heat-images", type=bool, default=False)
    parser.add_argument("-w", "--workers", type=int, default=0)
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("--base", type=str, default="ViT-H-14")
    parser.add_argument("--data-scale", type=float, default=1)
    parser.add_argument("--freeze-mode", type=str, default="a")
    parser.add_argument("-p", "--project", type=str, default="default")
    parser.add_argument("--lora-layers", type=str, nargs="+", default=[])
    parser.add_argument(
        "--precision",
        choices=["fp32", "amp", "fp16"],  # 新增 'fp32' / 'fp16'
        default="fp32"
    )

    parser.add_argument("--save-interval-epochs", type=int, default=1)
    return parser.parse_args()


def get_dataloaders(args, device, distributed: bool, rank: int, world: int):
    train_dataset = WenwuDataset(0, 0 + (0.8 * args.data_scale), args.memorize_images, device=device)
    val_dataset = WenwuDataset(0.8, 0.8 + (0.1 * args.data_scale), args.memorize_images, device=device)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset, num_workers=args.workers,
                              sampler=train_sampler,
                              persistent_workers=args.workers > 0,
                              prefetch_factor=4 if args.workers > 0 else None,
                              shuffle=train_sampler is None,
                              batch_size=args.batch_size,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset, num_workers=args.workers,
                            sampler=val_sampler,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=4 if args.workers > 0 else None,
                            shuffle=val_sampler is None,
                            batch_size=args.batch_size,
                            pin_memory=True)

    if args.heat_images:
        heat()

    return train_loader, val_loader, train_sampler, val_sampler


def lora(_model: CLIP):
    if args.lora_layers:
        # 1) 针对“文本 BERT”加 LoRA
        bert_lora = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"],
            bias="none"
        )
        print(_model.bert.config)
        _model.bert.config = _model.bert.config
        _model.bert = get_peft_model(_model.bert, bert_lora)

        # 2) 可选：视觉 Transformer 也加
        vit_lora = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["in_proj_weight", "out_proj", "c_fc"],
            bias="none"
        )
        _model.visual.transformer = get_peft_model(
            _model.visual.transformer, vit_lora
        )
        _model.bert.print_trainable_parameters()
        _model.visual.transformer.print_trainable_parameters()

    return model


if __name__ == "__main__":
    args = get_args()
    distributed, local_rank, world_size = setup_distributed()
    device = determine_device(args, local_rank)
    train_loader, val_loader, train_sampler, val_sampler = get_dataloaders(args, device, distributed, local_rank,
                                                                           world_size)
    use_amp = args.precision == "amp"
    scaler = torch.GradScaler(device=device, enabled=use_amp)

    model, preprocess = load_from_name(args.base, device=device, download_root='./base')
    model = move_to(model, device, args.precision)


    checkpoint = Path("checkpoints") / (args.project + ".pt")
    optimizer_state = None
    if args.project is not None and checkpoint.exists():
        model_state, optimizer_state, scaler_state, start_epoch = read_state(checkpoint, device)
        scaler.load_state_dict(scaler_state)
        model.load_state_dict(model_state)
    else:
        start_epoch = 1

    freeze(model, args.freeze_mode)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=False)
        model.encode_image = model.module.encode_image
        model.encode_text = model.module.encode_text

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    reports = read_reports(args.project)
    print(f"use_amp={use_amp} device={device} precision={args.precision}")

    for epoch in range(start_epoch, args.epochs + 1):

        if distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train(f"[{args.project}]{epoch}/{args.epochs} Training",
                           model=model,
                           train_loader=train_loader,
                           optimizer=optimizer,
                           device=device, precision=args.precision, use_amp=use_amp,
                           scaler=scaler)

        if local_rank != 0:
            continue

        eval_result = evaluate_clip_multicap(model, val_loader, device)
        ## 一些打印和保存结果的工作
        score = eval_result["score"]
        highest_score = 0
        for hitem in reports["history"]:
            s = hitem["performance"]["score"]
            if s > highest_score:
                highest_score = s

        reports["history"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "performance": eval_result,
        })
        # print(eval_result)
        save_reports(args.project, reports)

        if score > highest_score:
            highest_score = score
            save_state(filename=f"checkpoints/{args.project}.pt",
                       model=model, optimizer=optimizer, epoch=epoch, scaler=scaler)
