import os
import glob
import numpy as np
from PIL import Image

base_dir = "/home/groups/comp3710"

# Find image and mask files in the OASIS directory
mask_files = glob.glob(os.path.join(base_dir, "**", "*seg*.png"), recursive=True) + \
             glob.glob(os.path.join(base_dir, "**", "*mask*.png"), recursive=True)

if not mask_files:
    # Fallback search for any PNG files in the folder structure
    mask_files = glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True)

if not mask_files:
    print(f"No PNG files found under {base_dir}. Checking directory contents:")
    print(os.listdir(base_dir))
else:
    sample_mask_path = mask_files[0]
    print(f"Found mask file: {sample_mask_path}\n")

    # Load and inspect mask properties
    mask_img = Image.open(sample_mask_path)
    mask_array = np.array(mask_img)

    unique_classes = np.unique(mask_array)
    num_classes = len(unique_classes)

    print("--- Mask Metadata ---")
    print(f"Dimensions (H x W):       {mask_array.shape}")
    print(f"Data Type:               {mask_array.dtype}")
    print(f"Unique Label Values:     {unique_classes}")
    print(f"Total Output Classes (K): {num_classes}")