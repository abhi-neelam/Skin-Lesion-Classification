import os
import pandas as pd
import numpy as np
import shutil

train_data_folder_path = "../data/ISIC_2019_Training_Input"
train_annotations_file = "../data/ISIC_2019_Training_GroundTruth.csv"
test_data_folder_path = "../data/ISIC_2019_Test_Input"
test_annotations_file = "../data/ISIC_2019_Test_GroundTruth.csv"
out_dir = "../data/ISIC_2019_timm"

os.makedirs(out_dir, exist_ok=True) # create the output dir

train_img_labels = pd.read_csv(train_annotations_file)
test_img_labels = pd.read_csv(test_annotations_file)

train_samples = train_img_labels.sample(frac=0.80, random_state=42) # randomstate for consistency across objects
valid_samples = train_img_labels.drop(train_samples.index)
test_samples = test_img_labels

for split, samples_df, data_folder_path in zip(["train", "validation", "test"], [train_samples, valid_samples, test_samples], [train_data_folder_path, train_data_folder_path, test_data_folder_path]):
    samples_df = samples_df[samples_df['UNK'] == 0] # filter out unknown as we don't predict that category
    samples_df.drop(["UNK", "score_weight", "validation_weight"], axis=1, inplace=True, errors='ignore') # remove some unnecessary columns
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