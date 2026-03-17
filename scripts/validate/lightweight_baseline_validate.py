#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import csv
from torch import nn
from timm import utils
from timm.data import create_dataset
from timm.data.loader import create_loader
from timm.models import create_model, load_checkpoint
from timm.utils import reparameterize_model
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import OrderedDict
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

    def classifier(self, x):
        if hasattr(self.model, 'classifier'):
            x = self.model.classifier(x)
        else:
            x = self.model.head(x)
        return x

    def forward(self, x):
        return self.model(x)


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
                        help='Input batch size for validation (default: 32)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='Device (accelerator) to use.')

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
                        help='Enable confusion matrix summary '
                             'Requires matplotlib. (default: True)')

    parser.add_argument('--classification-report', action='store_true', default=True,
                        help='Enable confusion report summary '
                             'Requires scikit-learn. (default: True)')

    parser.add_argument('--tsne', action='store_true', default=True,
                        help='Enable tsne summary '
                             'Requires scikit-learn and seaborn. (default: False)')

    parser.add_argument('--throughput-warmup-iters', type=int, default=10,
                        help='Number of warmup iterations for throughput benchmarking (default: 10)')

    parser.add_argument('--throughput-iters', type=int, default=30,
                        help='Number of timed iterations for throughput benchmarking (default: 30)')

    args = parser.parse_args()

    return args


def is_cuda_device(device):
    return torch.device(device).type == 'cuda'


def synchronize_device(device):
    if is_cuda_device(device):
        torch.cuda.synchronize(device)


def clear_device_cache(device):
    if is_cuda_device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_oom_error(err):
    message = str(err).lower()
    oom_markers = (
        'out of memory',
        'cuda error: out of memory',
        'cublas_status_alloc_failed',
    )
    return any(marker in message for marker in oom_markers)


def forward_with_features(model, x):
    features = model.forward_head(x)
    output = model.classifier(features)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return features, output


def can_run_batch_size(model, device, input_size, batch_size):
    if batch_size < 1:
        return False

    dummy_input = None
    features = None
    output = None
    try:
        with torch.inference_mode():
            dummy_input = torch.randn((batch_size, *input_size), device=device)
            features, output = forward_with_features(model, dummy_input)
            synchronize_device(device)
        return True
    except (torch.OutOfMemoryError, RuntimeError) as err:
        if is_oom_error(err):
            return False
        raise
    finally:
        del output
        del features
        del dummy_input
        clear_device_cache(device)


def find_max_batch_size(model, device, input_size, starting_batch_size):
    starting_batch_size = max(1, int(starting_batch_size))

    if not is_cuda_device(device):
        return starting_batch_size

    hard_cap = max(starting_batch_size, 2048)

    if can_run_batch_size(model, device, input_size, starting_batch_size):
        best_batch_size = starting_batch_size
        probe_batch_size = min(starting_batch_size * 2, hard_cap)

        while probe_batch_size > best_batch_size and can_run_batch_size(model, device, input_size, probe_batch_size):
            best_batch_size = probe_batch_size
            if probe_batch_size >= hard_cap:
                break
            probe_batch_size = min(probe_batch_size * 2, hard_cap)

        failed_batch_size = probe_batch_size if probe_batch_size > best_batch_size else best_batch_size + 1
        low, high = best_batch_size + 1, failed_batch_size - 1
    else:
        best_batch_size = 0
        low, high = 1, starting_batch_size - 1

    while low <= high:
        mid = (low + high) // 2
        if can_run_batch_size(model, device, input_size, mid):
            best_batch_size = mid
            low = mid + 1
        else:
            high = mid - 1

    if best_batch_size < 1:
        raise RuntimeError('Unable to run inference even with batch size 1 on the selected device.')

    return best_batch_size


def benchmark_throughput(model, device, input_size, batch_size, warmup_iters=10, benchmark_iters=30):
    while batch_size >= 1:
        dummy_input = None
        features = None
        output = None
        try:
            dummy_input = torch.randn((batch_size, *input_size), device=device)
            with torch.inference_mode():
                for _ in range(max(0, warmup_iters)):
                    features, output = forward_with_features(model, dummy_input)
                    del output
                    del features
                    output = None
                    features = None

                synchronize_device(device)
                start_time = time.perf_counter()
                for _ in range(max(1, benchmark_iters)):
                    features, output = forward_with_features(model, dummy_input)
                    del output
                    del features
                    output = None
                    features = None
                synchronize_device(device)
                total_time = time.perf_counter() - start_time

            avg_latency = total_time / max(1, benchmark_iters)
            throughput = batch_size / avg_latency if avg_latency > 0 else 0.0
            return throughput, avg_latency, batch_size
        except (torch.OutOfMemoryError, RuntimeError) as err:
            if not is_oom_error(err):
                raise
            batch_size //= 2
        finally:
            del output
            del features
            del dummy_input
            clear_device_cache(device)

    raise RuntimeError('Unable to benchmark throughput even with batch size 1 on the selected device.')


def validate(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    input_size = (3, 224, 224)

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
    model.eval()

    throughput_batch_size = find_max_batch_size(model, device, input_size, args.batch_size)
    throughput, throughput_latency, throughput_batch_size = benchmark_throughput(
        model,
        device,
        input_size,
        throughput_batch_size,
        warmup_iters=args.throughput_warmup_iters,
        benchmark_iters=args.throughput_iters,
    )

    criterion = nn.CrossEntropyLoss().to(device)

    dataset = create_dataset(
        '',
        root=args.data_dir,
        split=args.split,
        class_map='',
        input_img_mode='RGB',
        input_key=None,
        target_key=None,
    )

    loader = create_loader(
        dataset,
        input_size=input_size,
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

    need_predictions = bool(args.metrics_avg or args.classification_report or args.confusion_matrix)
    need_features = bool(args.tsne)
    all_preds = [] if need_predictions else None
    all_targets = [] if need_predictions else None
    feature_chunks = [] if need_features else None

    data_start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            data_time.update(time.perf_counter() - data_start_time)
            batch_size = input.shape[0]

            input = input.to(device=device)
            target = target.to(device=device)

            synchronize_device(device)
            inference_start_time = time.perf_counter()

            features, output = forward_with_features(model, input)
            loss = criterion(output, target)
            topk = (1, min(5, args.num_classes))
            accuracies = utils.accuracy(output, target, topk=topk)
            acc1 = accuracies[0]
            acc5 = accuracies[-1] if len(accuracies) > 1 else accuracies[0]

            synchronize_device(device)
            inference_time.update(time.perf_counter() - inference_start_time)

            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            if need_features:
                feature_chunks.append(features.cpu())

            if need_predictions:
                predictions = torch.argmax(output, dim=1)
                all_preds.append(predictions.cpu())
                all_targets.append(target.cpu())

            data_start_time = time.perf_counter()

    feature_matrix = torch.cat(feature_chunks, dim=0).numpy() if need_features else None
    if need_predictions:
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

    top1a, top5a = top1.avg, top5.avg
    inference_time_avg = inference_time.avg
    data_time_avg = data_time.avg
    total_inference_time = inference_time.sum
    total_data_time = data_time.sum
    validation_throughput = top1.count / total_inference_time if total_inference_time > 0 else 0.0

    metric_results = {}
    if args.metrics_avg and need_predictions:
        precision = precision_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        recall = recall_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        metric_results = {
            f'{args.metrics_avg}_precision': round(100 * precision, 4),
            f'{args.metrics_avg}_recall': round(100 * recall, 4),
            f'{args.metrics_avg}_f1_score': round(100 * f1, 4),
        }

    labels = list(dataset.reader.class_to_idx.keys())

    if args.classification_report and need_predictions:
        print(classification_report(all_targets, all_preds, target_names=labels))
        report = classification_report(all_targets, all_preds, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose().rename_axis('label').reset_index()
        save_df(df_report, args, 'classification_report.csv')

    if args.confusion_matrix and need_predictions:
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        save_fig(args, 'confusion_matrix.png')

    if args.tsne and feature_matrix is not None and all_targets is not None:
        tsne = TSNE(n_components=2)
        tsne_data = tsne.fit_transform(feature_matrix)

        idx_to_class = {v: k for k, v in dataset.reader.class_to_idx.items()}
        target_labels = [idx_to_class[k] for k in all_targets.tolist()]

        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

        sns.scatterplot(
            x=tsne_data[:, 0], y=tsne_data[:, 1],
            hue=target_labels,
            legend='full',
            palette='bright',
            alpha=0.5,
            s=10,
            linewidth=0
        )
        ax.set_axis_off()
        save_fig(args, 'tsne.png')

    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        inference_time=round(inference_time_avg, 4), data_time=round(data_time_avg, 4),
        total_inference_time=round(total_inference_time, 4), total_data_time=round(total_data_time, 4),
        throughput=round(throughput, 2),
        throughput_batch_size=throughput_batch_size,
        throughput_latency=round(throughput_latency, 6),
        validation_throughput=round(validation_throughput, 2),
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
    plt.close()


def main():
    sns.set_palette('bright')

    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results = validate(args)

    p = Path(args.checkpoint)
    model_config_name = p.parent.name
    results_file = os.path.join(args.split, model_config_name, 'results.csv')
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
