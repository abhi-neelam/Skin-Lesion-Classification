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
from timm import utils
from timm.data import create_dataset
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
from functools import partial
from datetime import datetime
import pandas as pd
import numpy as np
import random
import time
import sys
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

def validate(model, loader_eval, validate_loss_fn, args, device):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)')

    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='mobilenetv3_large_100', type=str, metavar='MODEL',
                   help='Name of model to train (default: "mobilenetv3_large_100")')
    group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
    
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
    group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
    
    group = parser.add_argument_group('Device parameters')
    group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")

    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')

    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--lr', type=float, default=None, metavar='LR',
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
    group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
    group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
    group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
    group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
    group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
    
    group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
    group.add_argument('--wandb-project', default=None, type=str,
                    help='wandb project name')
    group.add_argument('--wandb-tags', default=[], type=str, nargs='+',
                    help='wandb tags')

    args = parser.parse_args()

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = parse_args()

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
    
    model.to(device=device, dtype=torch.float32)

    summary = summary(model, input_size=(args.batch_size, 3, 224, 224))
    macs = torchprofile.profile_macs(model, torch.randn(1, 3, 224, 224).to(device))

    print(summary)
    print(macs)

    optimizer = AdamW(lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    dataset_train = create_dataset(
        '',
        root=args.data_dir,
        split='train',
        is_training=True,
        class_map='',
        batch_size=args.batch_size,
        seed=args.seed,
        input_key=None,
        target_key=None, # TODO - remove this?
    )

    dataset_eval = create_dataset(
        '',
        root=args.data_dir,
        split='validation',
        is_training=False,
        class_map='',
        batch_size=args.batch_size,
        seed=args.seed,
        input_key=None,
        target_key=None, # TODO - remove this?
    )

    loader_train = create_loader(
        dataset_train,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=True,
        no_aug=True,
        num_workers=args.workers,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=False,
        no_aug=True,
        num_workers=args.workers,
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

    wandb.init(
        project=args.wandb_project,
        name=exp_name,
        config=args,
        tags=args.wandb_tags,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10, num_training_steps=args.epochs)

    for epoch in range(0, args.epochs):
        train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                device=device,
                lr_scheduler=lr_scheduler,
                output_dir=output_dir,
            )
        
        eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                )
        
        # TODO - UPDATE SUMMARY.CSV HERE

        saver.save_checkpoint(epoch)
        lr_scheduler.step(epoch + 1)

epochs_without_gain = 0
best_valid_loss = float("inf")

max_train_acc = 0
max_valid_acc = 0

num_batches_trained = 0
num_samples_trained = 0

print("Beginning training")
for epoch in range(start_epoch, epochs):
    train_epoch_loss, valid_epoch_loss = 0.0, 0.0

    train_epoch_preds, train_epoch_labels, train_epoch_probs = [], [], []
    valid_epoch_preds, valid_epoch_labels, valid_epoch_probs = [], [], []

    train_start_time = time.time()
    model.train()
    for batch_idx, (train_features, train_labels) in enumerate(dataloader_train):
        train_batch_size = len(train_labels)
        train_features = train_features.to(device)
        train_labels = train_labels.to(device) # move to device
        
        optimizer.zero_grad()

        predictions = model(train_features)
        probs = torch.softmax(predictions, dim=1)
        predictions_labels = torch.argmax(predictions, dim=1)

        train_epoch_preds.extend(predictions_labels.cpu().numpy())
        train_epoch_probs.extend(probs.detach().cpu().numpy())
        train_epoch_labels.extend(train_labels.cpu().numpy())

        train_batch_loss = loss(predictions, train_labels)
        train_batch_loss.backward()

        optimizer.step()

        train_epoch_loss += train_batch_loss.item() * train_batch_size

        num_batches_trained += 1
        num_samples_trained += train_batch_size

        train_batch_acc = accuracy_score(train_labels.cpu().numpy(), predictions_labels.cpu().numpy())

        model.eval()
        with torch.inference_mode():
            valid_features, valid_labels = next(iter(dataloader_valid)) # sample random validation mini batch
            valid_features = valid_features.to(device)
            valid_labels = valid_labels.to(device) # move to device

            predictions = model(valid_features)
            predictions_labels = torch.argmax(predictions, dim=1)

            valid_batch_loss = loss(predictions, valid_labels)
            valid_batch_acc = accuracy_score(valid_labels.cpu().numpy(), predictions_labels.cpu().numpy())
        model.train()

        writer.add_scalar("Loss/train-batch", train_batch_loss.item(), num_batches_trained)
        writer.add_scalar("Accuracy/train-batch", train_batch_acc, num_batches_trained)

        writer.add_scalar("Loss/valid-batch", valid_batch_loss.item(), num_batches_trained)
        writer.add_scalar("Accuracy/valid-batch", valid_batch_acc, num_batches_trained)
    train_end_time = time.time()

    valid_start_time = time.time()
    model.eval()
    with torch.inference_mode():
        for batch_idx, (valid_features, valid_labels) in enumerate(dataloader_valid):
            valid_batch_size = len(valid_labels)
            valid_features = valid_features.to(device)
            valid_labels = valid_labels.to(device) # move to device
            
            predictions = model(valid_features)
            probs = torch.softmax(predictions, dim=1)
            predictions_labels = torch.argmax(predictions, dim=1)

            valid_epoch_preds.extend(predictions_labels.cpu().numpy())
            valid_epoch_probs.extend(probs.detach().cpu().numpy())
            valid_epoch_labels.extend(valid_labels.cpu().numpy())

            valid_batch_loss = loss(predictions, valid_labels)
            valid_epoch_loss += valid_batch_loss.item() * valid_batch_size
    valid_end_time = time.time()

    train_epoch_loss /= len(train_dataset)
    valid_epoch_loss /= len(valid_dataset)

    # TODO - check if roc auc score is computed correctly.

    y_train_score = np.vstack(train_epoch_probs)
    y_train_onehot = label_binarize(train_epoch_labels, classes=np.arange(model.num_classes))

    y_valid_score = np.vstack(valid_epoch_probs)
    y_valid_onehot = label_binarize(valid_epoch_labels, classes=np.arange(model.num_classes))

    train_epoch_acc = accuracy_score(train_epoch_labels, train_epoch_preds)
    train_epoch_prec = precision_score(train_epoch_labels, train_epoch_preds, average='macro')
    train_epoch_rec = recall_score(train_epoch_labels, train_epoch_preds, average='macro')
    train_epoch_f1 = f1_score(train_epoch_labels, train_epoch_preds, average='macro')
    train_epoch_auc = roc_auc_score(y_train_onehot, y_train_score, average='macro', multi_class='ovr')
    train_epoch_spec = specificity_score(train_epoch_labels, train_epoch_preds, average='macro')

    valid_epoch_acc = accuracy_score(valid_epoch_labels, valid_epoch_preds)
    valid_epoch_prec = precision_score(valid_epoch_labels, valid_epoch_preds, average='macro')
    valid_epoch_rec = recall_score(valid_epoch_labels, valid_epoch_preds, average='macro')
    valid_epoch_f1 = f1_score(valid_epoch_labels, valid_epoch_preds, average='macro')
    valid_epoch_auc = roc_auc_score(y_valid_onehot, y_valid_score, average='macro', multi_class='ovr')
    valid_epoch_spec = specificity_score(valid_epoch_labels, valid_epoch_preds, average='macro')

    max_train_acc = max(max_train_acc, train_epoch_acc)
    max_valid_acc = max(max_valid_acc, valid_epoch_acc)

    writer.add_scalar("Loss/train-epoch", train_epoch_loss, epoch)
    writer.add_scalar('Accuracy/train-epoch', train_epoch_acc, epoch)
    writer.add_scalar('F1-Score/train-epoch', train_epoch_f1, epoch)
    writer.add_scalar('Precision/train-epoch', train_epoch_prec, epoch)
    writer.add_scalar('Recall-Sensitivity/train-epoch', train_epoch_rec, epoch)
    writer.add_scalar('AUC/train-epoch', train_epoch_auc, epoch)
    writer.add_scalar('Specificity/train-epoch', train_epoch_spec, epoch)
    writer.add_scalar('Elapsed Time/train-epoch-secs', train_end_time - train_start_time, epoch)

    writer.add_scalar("Loss/valid-epoch", valid_epoch_loss, epoch)
    writer.add_scalar('Accuracy/valid-epoch', valid_epoch_acc, epoch)
    writer.add_scalar('F1-Score/valid-epoch', valid_epoch_f1, epoch)
    writer.add_scalar('Precision/valid-epoch', valid_epoch_prec, epoch)
    writer.add_scalar('Recall-Sensitivity/valid-epoch', valid_epoch_rec, epoch)
    writer.add_scalar('AUC/valid-epoch', valid_epoch_auc, epoch)
    writer.add_scalar('Specificity/valid-epoch', valid_epoch_spec, epoch)
    writer.add_scalar('Elapsed Time/valid-epoch-secs', valid_end_time - valid_start_time, epoch)

    print(f"Saving epoch {epoch+1}...")
    torch.save(model.state_dict(), f"checkpoints/{model_name}/epoch-{epoch+1}.pth") # checkpoint per epoch for safety

    if valid_epoch_loss < best_valid_loss:
        best_valid_loss = valid_epoch_loss
        epochs_without_gain = 0

        torch.save(model.state_dict(), f"checkpoints/{model_name}/best_model.pth") # checkpoint the best model so far
    else:
        epochs_without_gain += 1

    if epochs_without_gain >= early_stopping_rounds:
        print(f"Early stopping at epoch {epoch+1}")
        break

print("Train Acc:", max_train_acc)
print("Valid Acc:", max_valid_acc)

writer.flush()


cm = confusion_matrix(valid_epoch_labels, valid_epoch_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.class_names)
disp.plot()

print(classification_report(valid_epoch_labels, valid_epoch_preds))

if __name__ == '__main__':
    main()