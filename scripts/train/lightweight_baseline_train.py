#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import wandb
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.profiler import profile, ProfilerActivity, record_function
from timm import utils
from timm.data import create_dataset, create_transform, resolve_data_config
from timm.data.loader import create_loader
from timm.models import create_model, safe_model_name
from torchtune.training import get_cosine_schedule_with_warmup
import torchprofile
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.metrics import sensitivity_score, specificity_score
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from functools import partial
from collections import OrderedDict
from datetime import datetime
from contextlib import nullcontext
import pandas as pd
import numpy as np
import random
import time
import os

class LightWeight_Baseline(nn.Module):
    def __init__(self, model, num_classes, pretrained=True, in_chans=3, dropout=0.2):
        super().__init__()

        self.model = create_model(
            model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=dropout,
        )

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.dropout = dropout

    def forward(self, x):
        x = self.model(x)
        return x

def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def experiment_name(args):
    if args.experiment:
        return args.experiment
    else:
        return '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            '224'
        ])

def train_one_epoch(args, model, loader, optimizer, loss_fn, device):
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    update_time_m = utils.AverageMeter()

    model.train()

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) if False else nullcontext()
    
    with prof:
        with record_function("model_inference"):
            for batch_idx, (input, target) in enumerate(loader):
                data_time_m.update(time.time() - data_start_time)
                
                batch_size = input.shape[0]

                input = input.to(device, non_blocking=True)
                target = target.to(device=device, non_blocking=True)

                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                loss.backward()
                optimizer.step()

                losses_m.update(loss.item(), batch_size)
                top1_m.update(acc1.item(), batch_size)
                top5_m.update(acc5.item(), batch_size)

                optimizer.zero_grad(set_to_none=True)

                update_time_m.update(time.time() - update_start_time)

                data_start_time = time.time()
                update_start_time = time.time()

    throughput = top1_m.count / update_time_m.sum
    
    metrics = OrderedDict([
        ('loss', losses_m.avg), 
        ('top1', top1_m.avg), 
        ('top5', top5_m.avg),
        ('data_time', data_time_m.avg),
        ('update_time', update_time_m.avg),
        ('total_data_time', data_time_m.sum),
        ('total_update_time', update_time_m.sum),
        ('throughput', throughput)
    ])
    
    return metrics

def validate(model, loader, loss_fn, device):
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    inference_time_m = utils.AverageMeter()

    model.eval()
    data_start_time = time.time()
    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            data_time_m.update(time.time() - data_start_time)
            batch_size = input.shape[0]

            inference_start_time = time.time()

            input = input.to(device=device)
            target = target.to(device=device)
            
            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            losses_m.update(loss.item(), batch_size)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

            inference_time_m.update(time.time() - inference_start_time)

            data_start_time = time.time()

    throughput = top1_m.count / inference_time_m.sum
    
    metrics = OrderedDict([
        ('loss', losses_m.avg),
        ('top1', top1_m.avg),
        ('top5', top5_m.avg),
        ('data_time_epoch', data_time_m.avg),
        ('inference_time_epoch', inference_time_m.avg),
        ('total_data_time', data_time_m.sum),
        ('total_inference_time', inference_time_m.sum),
        ('throughput', throughput)
    ])
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)', required=True)

    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', required=True, default='mobilenetv3_large_100', type=str, metavar='MODEL',
                   help='Name of model to train (default: "mobilenetv3_large_100")')
    group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--onlineaugment', action='store_true', default=False,
                   help='Online data augmentation procedure (default: False)')
    group.add_argument('--profile', action='store_true', default=False,
                   help='Enable profiling of cpu and torch functions (default: False)')
    
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)', required=True)
    group.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                   help='Input batch size for training (default: 32)')
    
    group = parser.add_argument_group('Device parameters')
    group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")

    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')

    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
    group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')

    group = parser.add_argument_group('Augmentation and regularization parameters')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
    group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
    
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
    group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
    group.add_argument('-j', '--workers', type=int, default=32, metavar='N',
                   help='how many training processes to use (default: 32)')
    group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)', required=True)
    group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output', required=True)
    
    group.add_argument('--wandb-project', default=None, type=str,
                    help='wandb project name', required=False)
    group.add_argument('--wandb-tags', default=[], type=str, nargs='*',
                    help='wandb tags', required=False)
    group.add_argument('--disable-wandb', action='store_true', default=False,
                help='Option to disable wandb logs to online')

    args = parser.parse_args()

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = parse_args()

    if args.profile:
        profiler = Profiler()
        profiler.start()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    random_seed(args.seed)

    device = torch.device(args.device)

    model = LightWeight_Baseline(
        args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        in_chans=3,
        dropout=args.drop,
        )
    
    model.to(device=device)

    model_summary = summary(model, input_size=(args.batch_size, 3, 224, 224))
    print(model_summary)

    unfrozen_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(unfrozen_params, lr=args.lr, weight_decay=args.weight_decay)

    dataset_train = create_dataset(
        '',
        root=args.data_dir,
        split='train',
        is_training=True,
        class_map='',
        batch_size=args.batch_size,
        seed=args.seed
    )

    dataset_eval = create_dataset(
        '',
        root=args.data_dir,
        split='validation',
        is_training=False,
        class_map='',
        batch_size=args.batch_size,
        seed=args.seed
    )

    loader_train = create_loader(
        dataset_train,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=True,
        no_aug=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        device=device,
        use_prefetcher=False,
    )
    
    args.rank = args.local_rank = 0
    data_cfg = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    base_train_transform = create_transform(input_size=(3,224,224), 
                                            is_training=True, 
                                            no_aug=True,
                                            mean=data_cfg['mean'], 
                                            std=data_cfg['std'])

    if args.onlineaugment:
        dataset_train.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomAffine(
                degrees=(45, 180),
                translate=(0.125, 0.125),
                scale=(0.90, 1.10)
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            base_train_transform
        ])

    loader_eval = create_loader(
        dataset_eval,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=False,
        no_aug=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        device=device
    )
    
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing).to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    exp_name = experiment_name(args)
    output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
    saver = utils.CheckpointSaver(
        model=model,
        optimizer=optimizer,
        args=args,
        checkpoint_dir=output_dir,
        max_history=args.checkpoint_hist
    )

    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=args,
            tags=args.wandb_tags,
        )

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_epochs, num_training_steps=args.epochs) # TODO - step per epoch or per batch?

    for epoch in range(0, args.epochs):
        train_metrics = train_one_epoch(
                args,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                device
            )
        
        eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    device,
                )
        
        utils.update_summary(
            epoch,
            train_metrics,
            eval_metrics,
            filename=os.path.join(output_dir, 'summary.csv'),
            log_wandb=not args.disable_wandb
        )

        saver.save_checkpoint(epoch, metric=eval_metrics['top1'])
        lr_scheduler.step()

    if args.profile:
        profiler.stop()
        profiler.open_in_browser()
        profiler.print()

if __name__ == '__main__':
    main()