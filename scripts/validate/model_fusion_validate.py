#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import csv
import yaml
import wandb
from torch import nn
from timm import utils
from timm.data import create_dataset
from timm.data.loader import create_loader
from timm.models import create_model, safe_model_name, load_checkpoint
from timm.utils import reparameterize_model
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, log_loss
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.metrics import sensitivity_score, specificity_score
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import random
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
    
    parser.add_argument('--classifier', default='lightgbm', type=str, metavar='CLF',
                   help='name of classifier to use for fusion (default: lightgbm, options - lightgbm, xgboost, hgbc, svc, logistic)')

    parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')

    parser.add_argument('--classifier-checkpoint', required=True, default='', type=str, metavar='PATH',
                    help='path to latest classifier checkpoint (default: none)')

    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output', required=False)
    
    parser.add_argument('--models', required=True, nargs='+', default='', type=str, metavar='MODEL',
                   help='Names of models to fuse (e.g: "mobilenetv3_large_100 resnet50")')

    parser.add_argument('--checkpoints', required=True, nargs='+', default='', type=str, metavar='PATH',
                    help='Paths to latest level 0 model checkpoints (default: none)')

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

    parser.add_argument('--normalize', action='store_true', default=True,
                   help='Normalize features before fusion')

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
    
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text

def get_pre_logits_and_labels(args, models, loader, device):
    for model in models:
        model.eval()

    features_np = []
    targets_np = []
    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            # batch_size = input.shape[0]

            input = input.to(device=device)
            target = target.to(device=device)

            fused_features = []
            for model in models:
                features = model.forward_head(input) # (batch_size, feature_count)
                
                if args.normalize:
                    features = nn.functional.normalize(features, dim=1) # normalize each model feature vector

                fused_features.append(features)

            fused_features = torch.cat(fused_features, dim=1)
            features_np.append(fused_features.cpu())
            targets_np.append(target.cpu())
    features_np = torch.cat(features_np, dim=0).numpy()
    targets_np = torch.cat(targets_np, dim=0).numpy()
    
    X = features_np
    y = targets_np
    
    assert np.isfinite(X).all()
    
    cols = [f"Column_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=cols)
    
    return X, y

def validate(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    models = []
    for i, (model_str, checkpoint) in enumerate(zip(args.models, args.checkpoints)):
        model = LightWeight_Baseline(model_str, num_classes=args.num_classes, in_chans=3)
        load_checkpoint(model, checkpoint, use_ema=False, strict=False)
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
        batch_size=args.batch_size,
        input_key=None,
        target_key=None
    )
    
    loader_eval = create_loader(
        dataset,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        no_aug=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        device=device
    )

    clf = joblib.load(args.classifier_checkpoint)
    
    fusion_start_time = time.time()
    X_valid, y_valid = get_pre_logits_and_labels(args, models, loader_eval, device)
    fusion_inference_time = time.time() - fusion_start_time
    
    classifier_start_time = time.time()
    valid_predictions = clf.predict(X_valid)
    
    total_inference_time = time.time() - fusion_start_time
    classifier_inference_time = time.time() - classifier_start_time
    
    metric_results = {}
    if args.metrics_avg:
        precision = precision_score(y_valid, valid_predictions, average=args.metrics_avg, zero_division=0)
        recall = recall_score(y_valid, valid_predictions, average=args.metrics_avg, zero_division=0)
        f1 = f1_score(y_valid, valid_predictions, average=args.metrics_avg, zero_division=0)
        metric_results = {
            f'{args.metrics_avg}_precision': round(100 * precision, 4),
            f'{args.metrics_avg}_recall': round(100 * recall, 4),
            f'{args.metrics_avg}_f1_score': round(100 * f1, 4),
        }

    labels = list(dataset.reader.class_to_idx.keys()) # this should be sorted

    if args.classification_report:
        print(classification_report(y_valid, valid_predictions, target_names=labels))
        report = classification_report(y_valid, valid_predictions, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose().rename_axis("label").reset_index()
        save_df(df_report, args, "classification_report.csv")

    if args.confusion_matrix:
        cm = confusion_matrix(y_valid, valid_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        save_fig(args, "confusion_matrix.png")

    if args.tsne:
        tsne = TSNE(n_components=2)
        tsne_data = tsne.fit_transform(X_valid)

        idx_to_class = {v: k for k, v in dataset.reader.class_to_idx.items()}
        target_labels = [idx_to_class[k] for k in y_valid.tolist()]

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        sns.scatterplot(
            x=tsne_data[:,0], y=tsne_data[:,1],
            hue=target_labels,
            legend="full",
            palette="bright",
            alpha=0.3
        )
        save_fig(args, "tsne.png")
        plt.close(fig)

    accuracy = accuracy_score(y_valid, valid_predictions)
    throughput = len(X_valid) / classifier_inference_time

    results = OrderedDict(
        classifier=args.classifier,
        accuracy=round(accuracy, 4),
        fusion_inference_time=round(fusion_inference_time, 4),
        classifier_inference_time=round(classifier_inference_time, 4),
        total_inference_time=round(total_inference_time, 4),
        throughput=round(throughput),
        **metric_results
    )
    
    if args.classifier != "svc":
        valid_proba = clf.predict_proba(X_valid)
        
        results['log_loss'] = log_loss(y_valid, valid_proba)
    
    print(results)

    return results

def save_df(df, args, filename):
    p = Path(args.classifier_checkpoint)
    model_config_name = p.parent.name
    os.makedirs(model_config_name, exist_ok=True)
    output_path = os.path.join(model_config_name, filename)
    df.to_csv(output_path, index=True)

def save_fig(args, filename):
    p = Path(args.classifier_checkpoint)
    model_config_name = p.parent.name
    os.makedirs(model_config_name, exist_ok=True)
    plt.savefig(os.path.join(model_config_name, filename))

def main():
    sns.set_palette("bright")

    args, args_text = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results = validate(args)
    
    p = Path(args.classifier_checkpoint)
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