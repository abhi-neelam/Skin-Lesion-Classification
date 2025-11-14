import os
import pandas as pd
import numpy as np
import shutil

train_data_folder_path = "../data/ISIC_2019_Training_Input"
train_annotations_file = "../data/ISIC_2019_Training_GroundTruth.csv"
out_dir = "../data/ISIC_2019_timm"

os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(train_annotations_file)
df = df[df["UNK"] == 0].drop(["UNK", "score_weight", "validation_weight"], axis=1, errors="ignore")

img_col = df.columns[0]
label_cols = df.columns[1:]

train = df.sample(frac=0.80, random_state=42)
rem = df.drop(train.index)
valid = rem.sample(frac=0.50, random_state=42)
test = rem.drop(valid.index)

for split_name, split_df in [("train", train), ("validation", valid), ("test", test)]:
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for c in label_cols:
        os.makedirs(os.path.join(split_dir, c), exist_ok=True)

    label_idx = split_df[label_cols].to_numpy().argmax(axis=1)
    labels = [label_cols[i] for i in label_idx]

    for name, lbl in zip(split_df[img_col], labels):
        src = os.path.join(train_data_folder_path, f"{name}.jpg")
        dst = os.path.join(split_dir, lbl, f"{name}.jpg")
        shutil.copyfile(src, dst)