import os
import shutil
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

input_dir = "../data/ISIC_2019_timm"
output_dir = "../data/ISIC_2019_timm_augmented"

shutil.copytree(input_dir, output_dir, dirs_exist_ok=True) # copy entire ISIC 2019 directory

transforms = v2.Compose([
    v2.RandomAffine(
        degrees=(45, 180),
        translate=(0.125, 0.125),
        scale=(0.90, 1.10),
        interpolation=InterpolationMode.BILINEAR
    ),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ColorJitter(brightness=0.20, contrast=0.15, saturation=0.10),
])

aug_mults = {'AK': 5, 'DF': 5, 'VASC': 5, 'SCC': 5, 'BKL': 3, 'MEL': 2, 'BCC': 2}
for category, mult in aug_mults.items():
    in_dir = f"{input_dir}/train/{category}"
    out_dir = f"{output_dir}/train/{category}"

    for fname in os.listdir(in_dir):
        in_image = f"{in_dir}/{fname}"
        stem, ext = os.path.splitext(fname)
        img = Image.open(in_image).convert("RGB")

        for i in range(mult-1):
            transforms(img).save(f"{out_dir}/{stem}_aug{i+1}{ext}")