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

class LightWeight_Baseline(nn.Module):
    def __init__(self, model, num_classes, in_chans=3):
        super().__init__()

        self.model = create_model(
            model,
            in_chans=in_chans,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
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

    parser.add_argument('--model', required=True, default='mobilenetv3_large_100', type=str, metavar='MODEL',
                   help='Name of model to train (default: "mobilenetv3_large_100")')
    
    parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')

    parser.add_argument('--checkpoint', required=True, default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

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
    
    parser.add_argument('--metrics-avg', type=str, default=None,
                    choices=['micro', 'macro', 'weighted'],
                    help='Enable precision, recall, F1-score calculation and specify the averaging method. '
                         'Requires scikit-learn. (default: None)')

    parser.add_argument('--confusion-matrix', action='store_true', default=True,
                    help='Enable confusion matrix summary'
                         'Requires matplotlib. (default: True)')
    
    parser.add_argument('--classification-report', action='store_true', default=True,
                    help='Enable confusion report summary'
                         'Requires scikit-learn. (default: True)')

    parser.add_argument('--tsne', action='store_true', default=True,
                    help='Enable tsne summary'
                         'Requires scikit-learn and seaborn. (default: False)')
    
    args = parser.parse_args()

    return args

def validate(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    model = LightWeight_Baseline(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
    )

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, False)

    if args.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    
    dataset = create_dataset(
        '',
        root=args.data_dir,
        split=args.split,
        class_map='',
        input_img_mode='RGB',
        input_key=None,
        target_key=None, # TODO - remove this?
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
    data_time = utils.AverageMeter()
    inference_time = utils.AverageMeter()
    
    if args.metrics_avg:
        all_preds = []
        all_targets = []

    model.eval()
    data_start_time = time.time()
    with torch.inference_mode():
        feature_chunks = []
        for batch_idx, (input, target) in enumerate(loader):
            data_time.update(time.time() - data_start_time)
            batch_size = input.shape[0]

            inference_start_time = time.time()

            input = input.to(device=device)
            target = target.to(device=device)

            features = model.forward_head(input)
            feature_chunks.append(features.cpu())

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            inference_time.update(time.time() - inference_start_time)

            data_start_time = time.time()
            
            if args.metrics_avg:
                predictions = torch.argmax(output, dim=1)
                all_preds.append(predictions.cpu())
                all_targets.append(target.cpu())

        feature_matrix = torch.cat(feature_chunks, dim=0).numpy()

    top1a, top5a = top1.avg, top5.avg
    inference_time, data_time, total_inference_time, total_data_time = inference_time.avg, data_time.avg, inference_time.sum, data_time.sum
    throughput = top1.count / total_inference_time

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    metric_results = {}
    if args.metrics_avg:
        precision = precision_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        recall = recall_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        metric_results = {
            f'{args.metrics_avg}_precision': round(100 * precision, 4),
            f'{args.metrics_avg}_recall': round(100 * recall, 4),
            f'{args.metrics_avg}_f1_score': round(100 * f1, 4),
        }

    labels = list(dataset.reader.class_to_idx.keys()) # this should be sorted

    if args.classification_report:
        print(classification_report(all_targets, all_preds, target_names=labels))
        report = classification_report(all_targets, all_preds, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose().rename_axis("label").reset_index()
        save_df(df_report, args, "classification_report.csv")

    if args.confusion_matrix:
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        save_fig(args, "confusion_matrix.png")

    if args.tsne:
        tsne = TSNE(n_components=2)
        tsne_data = tsne.fit_transform(feature_matrix)

        idx_to_class = {v: k for k, v in dataset.reader.class_to_idx.items()}
        target_labels = [idx_to_class[k] for k in all_targets.tolist()]

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        sns.scatterplot(
            x=tsne_data[:,0], y=tsne_data[:,1],
            hue=target_labels,
            legend="full",
            palette="bright",
            alpha=0.3
        )
        save_fig(args, "tsne.png")

    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        inference_time=round(inference_time, 4), data_time=round(data_time, 4),
        total_inference_time=round(total_inference_time, 4), total_data_time=round(total_data_time, 4),
        throughput=round(throughput),
        **metric_results,
        param_count=round(param_count / 1e6, 2),
    )
    
    print(results)

    return results

def save_df(df, args, filename):
    p = Path(args.checkpoint)
    model_config_name = p.parent.name
    os.makedirs(os.path.join(args.split, model_config_name), exist_ok=True)
    output_path = os.path.join(args.split, model_config_name, filename)
    df.to_csv(output_path, index=True)

def save_fig(args, filename):
    p = Path(args.checkpoint)
    model_config_name = p.parent.name
    os.makedirs(os.path.join(args.split, model_config_name), exist_ok=True)
    plt.savefig(os.path.join(args.split, model_config_name, filename))

def main():
    sns.set_palette("bright")

    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results = validate(args)

    p = Path(args.checkpoint)
    model_config_name = p.parent.name
    results_file = os.path.join(args.split, model_config_name, "results.csv")
    os.makedirs(os.path.join(args.split, model_config_name), exist_ok=True)
    write_results(results_file, results)

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