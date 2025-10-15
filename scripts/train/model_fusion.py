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
import torchprofile
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.metrics import sensitivity_score, specificity_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
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

def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def experiment_name(args):
    if args.experiment:
        return args.experiment
    name = "+".join(args.models)[:60].replace("/", "_")
    return "-".join([datetime.now().strftime("%Y%m%d-%H%M%S"), name, "224"])
        
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

    parser.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)', required=True)

    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output', required=False)
    
    parser.add_argument('--models', required=True, nargs='+', default='', type=str, metavar='MODEL',
                   help='Names of models to fuse (e.g: "mobilenetv3_large_100 resnet50")')

    parser.add_argument('--checkpoints', required=True, nargs='+', default='', type=str, metavar='PATH',
                    help='Paths to latest checkpoints (default: none)')

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
    
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
    
    parser.add_argument('--weight-decay', type=float, default=1.0,
                    help='weight decay (default: 1.0)')
    
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: 0.05)')
    
    parser.add_argument('--trees', type=int, default=1000, metavar='T',
                   help='number of boosted trees to fit (default: 300)')
    
    parser.add_argument('--max-depth', type=int, default=10, metavar='D',
                   help='max tree depth (default: 10)')
    
    parser.add_argument('--wandb-project', default=None, type=str,
                    help='wandb project name', required=False)
    
    parser.add_argument('--wandb-tags', default=[], type=str, nargs='*',
                    help='wandb tags', required=False)
    
    args = parser.parse_args()
    
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text

def get_pre_logits_and_labels(models, loader, device):
    for model in models:
        model.eval()

    features_np = []
    targets_np = []
    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            if batch_idx > 3: # TODO - remove after
                break
            # batch_size = input.shape[0]

            input = input.to(device=device)
            target = target.to(device=device)

            fused_features = []
            for model in models:
                features = model.forward_head(input) # (batch_size, feature_count)
                features = nn.functional.normalize(features, dim=1) # normalize each model feature vector
                fused_features.append(features)

            fused_features = torch.cat(fused_features, dim=1)
            features_np.append(fused_features.cpu())
            targets_np.append(target.cpu())
    features_np = torch.cat(features_np, dim=0).numpy()
    targets_np = torch.cat(targets_np, dim=0).numpy()
    
    X = features_np
    y = targets_np
    
    cols = [f"Column_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=cols)
    
    return X, y

def train(args):
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

    dataset_train = create_dataset(
        '',
        is_training=True,
        root=args.data_dir,
        split='train',
        class_map='',
        batch_size=args.batch_size,
        seed=args.seed,
        input_key=None,
        target_key=None
    )
    
    dataset_eval = create_dataset(
        '',
        root=args.data_dir,
        split='validation',
        class_map='',
        batch_size=args.batch_size,
        seed=args.seed,
        input_key=None,
        target_key=None
    )

    loader_train = create_loader(
        dataset_train,
        is_training=True,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        no_aug=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        device=device
    )
    
    loader_eval = create_loader(
        dataset_eval,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        no_aug=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        device=device
    )

    X_train, y_train = get_pre_logits_and_labels(models, loader_train, device)
    X_valid, y_valid = get_pre_logits_and_labels(models, loader_eval, device)

    eval_result = {}
    
    clf = LGBMClassifier(num_class=args.num_classes, n_estimators=args.trees, max_depth=args.max_depth, objective='multiclass', device_type="gpu", verbosity=-1, n_jobs=args.workers, random_state=args.seed, learning_rate=args.lr, reg_lambda=args.weight_decay)
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="multi_logloss",
        callbacks=[
            lgb.log_evaluation(),
            lgb.record_evaluation(eval_result),
        ]
    )
    
    predictions = clf.predict(X_valid)
    print("Valid accuracy:", accuracy_score(y_valid, predictions))

    return clf

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

    args, args_text = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    exp_name = experiment_name(args)
    
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=args,
            tags=args.wandb_tags,
        )

    output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
    random_seed(args.seed)
    clf = train(args)
    
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
        
    joblib.dump(clf, os.path.join(output_dir, "lgbm.pkl"))

    if args.wandb_project:
        wandb.finish()

if __name__ == '__main__':
    main()