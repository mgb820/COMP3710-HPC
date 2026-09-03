import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

RAW_LABELS = np.array([
    0, 2, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 33, 34,
    35, 39, 72, 76, 77, 78, 84, 85, 88, 89, 113, 115, 116, 117, 126, 127,
    128, 129, 138, 139, 140, 166, 167, 170, 171, 177, 179, 183, 210, 216,
    217, 219, 220, 221, 222, 227, 228, 229, 231, 232, 233, 234, 235, 244,
    245, 246, 255
])
LABEL_MAP = {val: idx for idx, val in enumerate(RAW_LABELS)}

class OASISDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 256), one_hot=True):
        self.img_dir = os.path.join(root_dir, f"keras_png_slices_{split}")
        self.seg_dir = os.path.join(root_dir, f"keras_png_slices_seg_{split}")
        self.img_size = img_size
        self.one_hot = one_hot

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.seg_paths = sorted(glob.glob(os.path.join(self.seg_dir, "*.png")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load & normalize MRI input [1, H, W]
        img = Image.open(self.img_paths[idx]).convert("L").resize(self.img_size, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

        # 2. Load & resize mask using NEAREST neighbor to preserve integer IDs
        mask = Image.open(self.seg_paths[idx]).resize(self.img_size, Image.NEAREST)
        mask_arr = np.array(mask, dtype=np.int64)

        # 3. Remap raw pixel values to 0..64
        remapped = np.zeros_like(mask_arr, dtype=np.int64)
        for raw_val, mapped_idx in LABEL_MAP.items():
            remapped[mask_arr == raw_val] = mapped_idx

        target = torch.from_numpy(remapped)

        # 4. Optional One-Hot Encoding [65, H, W]
        if self.one_hot:
            target = torch.nn.functional.one_hot(target, num_classes=len(RAW_LABELS))
            target = target.permute(2, 0, 1).float()

        return img_tensor, target

if __name__ == "__main__":
    dataset = OASISDataset(root_dir="/home/groups/comp3710/OASIS", split="train")
    img, target = dataset[0]
    print("--- Dataset Output Check ---")
    print(f"Image tensor shape:  {img.shape}")      # Expected: torch.Size([1, 256, 256])
    print(f"Target tensor shape: {target.shape}")   # Expected: torch.Size([65, 256, 256])