import os
import pandas as pd
import numpy as np
import shutil

ADD_ISIC_2019 = True
ADD_DERM12345 = True

isic_train_data_folder_path = "../data/ISIC_2019_Training_Input"
isic_train_annotations_file = "../data/ISIC_2019_Training_GroundTruth.csv"

derm_train_data_folder_path = "../data/derm12345/images"
derm_train_annotations_file = "../data/derm12345/derm12345_supplemental.csv"

out_dir = "../data/ISIC_2019_derm12345_timm"

os.makedirs(out_dir, exist_ok=True) # create the output dir

derm_isic_class_mapping_dict = {
  "MEL": ["anm", "alm", "lm", "lmm", "mel"],
  "NV":  ["acb","ccb","mcb","cb","bdb","db","ajb","cjb","jb","acd","ccd","cd","ajd","srjd","jd","rd"],
  "BCC": ["bcc"],
  "AK":  ["ak", "bd"],
  "BKL": ["sk", "sl", "lk"],
  "DF":  ["df"],
  "VASC":["angk","ha","la","pg","sa"],
  "SCC": ["scc"]
}

leaf_to_isic = {leaf: k for k, vs in derm_isic_class_mapping_dict.items() for leaf in vs}

def safe_copy(src_path, dst_path):
    base, ext = os.path.splitext(dst_path)
    cand = dst_path
    k = 1
    while os.path.exists(cand):
        cand = f"{base}_{k}{ext}"
        k += 1
    shutil.copyfile(src_path, cand)

def norm_split(s):
    s = str(s).strip().lower()
    if s in ["train", "training", "tr"]:
        return "train"
    if s in ["validation", "val", "valid", "dev"]:
        return "validation"
    if s in ["test", "testing", "te"]:
        return "test"
    return None

if ADD_ISIC_2019:
    train_img_labels = pd.read_csv(isic_train_annotations_file)

    train_samples = train_img_labels.sample(frac=0.80, random_state=42) # randomstate for consistency across objects
    remainder = train_img_labels.drop(train_samples.index)
    valid_samples = remainder.sample(frac=0.50, random_state=42) # 10% for validation
    test_samples = remainder.drop(valid_samples.index) # 10% for test

    for split, samples_df in zip(["train", "validation", "test"], [train_samples, valid_samples, test_samples]):
        samples_df = samples_df[samples_df['UNK'] == 0] # filter out unknown as we don't predict that category
        samples_df = samples_df.drop(["UNK", "score_weight", "validation_weight"], axis=1, errors='ignore') # remove some unnecessary columns
        samples_df = samples_df.reset_index(drop=True) # reset index for stability
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
            safe_copy(
                f"{isic_train_data_folder_path}/{name}.jpg",
                f"{out_dir}/{split}/{col_name}/{name}.jpg"
            ) # copy image to directory

if ADD_DERM12345:
    derm_df = pd.read_csv(derm_train_annotations_file) # read derm annotations file

    for _, row in derm_df.iterrows():
        split = norm_split(row["split"])
        if split is None:
            continue
        leaf = str(row["label"]).strip()
        if leaf not in leaf_to_isic:
            continue
        col_name = leaf_to_isic[leaf]
        isic_id = str(row["isic_id"]).strip()
        os.makedirs(f"{out_dir}/{split}/{col_name}", exist_ok=True)
        dst_name = f"derm_{isic_id}.jpg"
        safe_copy(
            f"{derm_train_data_folder_path}/{isic_id}.jpg",
            f"{out_dir}/{split}/{col_name}/{dst_name}"
        ) # copy image to directory