import os
import sys
import pandas as pd
import numpy as np
import shutil

num_classes = 8

data_folder_path = "../data/ISIC_2019_Training_Input"
annotations_file = "../data/ISIC_2019_Training_GroundTruth.csv"
out_dir = "ISIC_2019"

os.makedirs(out_dir, exist_ok=True) # create the output dir

img_labels = pd.read_csv(annotations_file)

train_samples = img_labels.sample(frac=0.80, random_state=42) # randomstate for consistency across objects
valid_samples = img_labels.drop(train_samples.index)

for split, samples_df in zip(["train", "validation"], [train_samples, valid_samples]):
    samples_df.drop("UNK", axis=1, inplace=True) # remove unknown category
    samples_df.reset_index(drop=True, inplace=True) # reset index for stability
    ohe_labels = samples_df.iloc[:, 1:]

    labels = np.where(ohe_labels==1)[1]

    os.makedirs(f"{out_dir}/{split}", exist_ok=True) # create the split dir

    column_names = ohe_labels.columns.to_list()
    for col_name in column_names:
        os.makedirs(f"{out_dir}/{split}/{col_name}", exist_ok=True) # create the class directories

    for i, tup in samples_df.iterrows():
        name = tup[0]
        label = labels[i]
        col_name = column_names[label]

        shutil.copyfile(f"{data_folder_path}/{name}.jpg", f"{out_dir}/{split}/{col_name}/{name}.jpg")