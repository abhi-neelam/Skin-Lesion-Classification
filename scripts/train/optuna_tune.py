import os
import shutil
import subprocess
from pathlib import Path

import argparse
import optuna
import pandas as pd

WEIGHT_DECAYS = [0.005, 0.01, 0.02, 0.03, 0.05]
LABEL_SMOOTH = [0.05, 0.1, 0.2, 0.3]
LOSSES = ["cross_entropy", "weighted_cross_entropy", "focal"]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def parse_args():
    parser = argparse.ArgumentParser(description='Optuna Tuning')

    group = parser.add_argument_group('Optuna parameters')
    group.add_argument('--trials', type=int, default=20, metavar='N',
                   help='number of optuna trials (default: 20)')

    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)', required=True)

    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', required=True, default='mobilenetv3_large_100', type=str, metavar='MODEL',
                   help='Name of model to train (default: "mobilenetv3_large_100")')
    
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)', required=True)
    group.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                   help='Input batch size for training (default: 32)')
    
    group = parser.add_argument_group('Device parameters')
    group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")

    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
    
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
    group.add_argument('-j', '--workers', type=int, default=32, metavar='N',
                   help='how many training processes to use (default: 32)')
    
    group.add_argument('--wandb-tags', default=[], type=str, nargs='*',
                    help='wandb tags', required=False)
    group.add_argument('--disable-wandb', action='store_true', default=False,
                help='Option to disable wandb logs to online')

    args = parser.parse_args()

    return args

args = parse_args()

def objective(trial: optuna.Trial) -> float:
    wd = trial.suggest_categorical("weight_decay", WEIGHT_DECAYS)
    ls = trial.suggest_categorical("label_smoothing", LABEL_SMOOTH)
    loss_name = trial.suggest_categorical("loss", LOSSES)
    drop = trial.suggest_categorical("dropout", DROPOUTS)

    out_root = Path("./optuna_logs")
    out_root.mkdir(parents=True, exist_ok=True)

    exp_name = (
        f"optuna_t{trial.number:03d}_{args.model}_"
        f"wd{wd:.4g}_ls{ls:.3g}_loss{loss_name}_drop{drop:.1f}"
    )

    trial_out = out_root
    trial_dir = trial_out / exp_name

    if trial_dir.exists():
        shutil.rmtree(trial_dir)

    cmd = [
        "python", "lightweight_baseline_train.py",
        "--data-dir", args.data_dir,
        "--model", args.model,
        "--num-classes", str(args.num_classes),
        "--loss", loss_name,
        "--weight-decay", str(wd),
        "--smoothing", str(ls),
        "--drop", str(drop),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--pretrained",
        "--onlineaugment",
        "--pin-mem",
        "-b", str(args.batch_size),
        "--workers", str(args.workers),
        "--device", args.device,
        "--seed", str(args.seed),
        "--output", str(trial_out),
        "--experiment", exp_name,
        "--wandb-project", "skin-lesion-classification",
        "--wandb-tags", "optuna", "baseline", f"{args.model}", "pretrained", "finetuned", "isic_2019", "derm12345", "unfrozen_layers", "onfly_augmentation", *args.wandb_tags
    ]

    if args.disable_wandb:
        cmd.append("--disable-wandb")

    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        trial.set_user_attr("cmd", " ".join(cmd))
        trial.set_user_attr("stderr_tail", result.stderr[-4000:])
        trial.set_user_attr("stdout_tail", result.stdout[-4000:])
        raise RuntimeError(f"Training failed for trial {trial.number}")

    summary_csv = trial_dir / "summary.csv"
    df = pd.read_csv(summary_csv)
    best_eval_top1 = float(df["eval_top1"].max())

    return best_eval_top1

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5,
        seed=args.seed
    )

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)