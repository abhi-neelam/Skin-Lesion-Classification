#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import csv
from torch import nn
from timm import utils
from timm.data import create_dataset
from timm.data.loader import create_loader
from timm.models import create_model, safe_model_name, load_checkpoint
from timm.utils import reparameterize_model
import torchprofile
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.metrics import sensitivity_score, specificity_score
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import time
import os

def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
class LightWeight_Baseline(nn.Module):
    def __init__(self, model, num_classes, pretrained=True, in_chans=3):
        super().__init__()

        self.model = create_model(
            model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.flatten = nn.Flatten()

    def forward_head(self, x):
        x = self.model.forward_features(x)
        return self.model.forward_head(x, pre_logits=True)

    def forward(self, x):
        x = self.model(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Validation')

    parser.add_argument('--models', required=True, nargs='+', default='', type=str, metavar='MODEL',
                   help='Names of models to fuse (e.g: "mobilenetv3_large_100 resnet50")')
    
    parser.add_argument('--split', metavar='NAME', default='train',
                    help='dataset split (default: train)')

    parser.add_argument('--checkpoints', required=True, nargs='+', default='', type=str, metavar='PATH',
                    help='Paths to latest checkpoints (default: none)')

    parser.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
    
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)', required=True)
    
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                   help='Input batch size for training (default: 32)')
    
    parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
    
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
    
    parser.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')

    parser.add_argument('--data-dir', required=True, metavar='DIR',
                    help='path to dataset (root dir)')
    
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                   help='number of epochs to train (default: 100)')
    
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
    
    args = parser.parse_args()

    return args

def train(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    models = []
    for i, (model_str, checkpoint) in enumerate(zip(args.models, args.checkpoints)):
        model = LightWeight_Baseline(model_str, num_classes=args.num_classes, pretrained=args.pretrained, in_chans=3)
        load_checkpoint(model, checkpoint, False)
        if args.reparam:
            model = reparameterize_model(model)
    
        param_count = sum([m.numel() for m in model.parameters()])
        print('Model %s created, param count: %d' % (model_str, param_count))
        
        model.to(device=device)
        models.append(model)

    dataset = create_dataset(
        '',
        root=args.data_dir,
        split=args.split,
        class_map='',
        seed=args.seed,
        input_img_mode='RGB',
        input_key=None,
        target_key=None
    )

    loader = create_loader(
        dataset,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        no_aug=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        device=device
    )

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for model in models:
        model.eval()

    features_np = []
    targets_np = []
    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            batch_size = input.shape[0]

            input = input.to(device=device)
            target = target.to(device=device)

            fused_features = []
            for model in models:
                features = model.forward_head(input) # (batch_size, feature_count)
                features = nn.functional.normalize(features, dim=1) # normalize each model feature vector
                fused_features.append(features)

            fused_features = torch.cat(fused_features, dim=1)
            features_np.append(fused_features)
            targets_np.append(target)
    features_np = torch.cat(features_np, dim=0).cpu().numpy()
    targets_np = torch.cat(targets_np, dim=0).cpu().numpy()
    
    df = pd.DataFrame(features_np)
    df["target"] = targets_np



    return None

def save_df(df, args, filename):
    p = Path(args.checkpoint)
    model_config_name = p.parent.name
    os.makedirs(model_config_name, exist_ok=True)
    output_path = os.path.join(model_config_name, filename)
    df.to_csv(output_path, index=True)

def save_fig(args, filename):
    p = Path(args.checkpoint)
    model_config_name = p.parent.name
    os.makedirs(model_config_name, exist_ok=True)
    plt.savefig(os.path.join(model_config_name, filename))

def main():
    sns.set_palette("bright")

    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    random_seed(args.seed)
    results = train(args)

    if args.results_file:
        write_results(args.results_file, results)
    
def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        if not isinstance(results, (list, tuple)):
            results = [results]
        if not results:
            return
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()

if __name__ == '__main__':
    main()