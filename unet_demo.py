import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet

# 1. Device Setup & Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "unet_oasis.pth"  # Path to your saved weights
NUM_CLASSES = 65  # Target anatomical classes for OASIS

# 2. Dice Score Evaluation Function
def compute_dsc(pred, target, num_classes=NUM_CLASSES, eps=1e-7):
    """Calculates Mean Dice Similarity Coefficient across all classes."""
    pred_onehot = torch.nn.functional.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    
    dsc_per_class = (2.0 * intersection + eps) / (union + eps)
    return dsc_per_class.mean().item()

# 3. Load Trained Model
print(f"--- OASIS UNet Live Demonstration ---")
print(f"Running on Device: {device}")

model = UNet(in_channels=1, out_channels=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

from torch.utils.data import DataLoader
from unet_dataset import OASISDataset

test_dataset = OASISDataset(img_dir="/path/to/keras_png_slices_test")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 4. Fetch Demonstration Sample
# Replace 'test_loader' with your active OASIS DataLoader instance
inputs, targets = next(iter(test_loader))
inputs, targets = inputs.to(device), targets.to(device)

# 5. Execute Synchronized Live Inference
start_time = time.time()
with torch.no_grad():
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
    else:
        outputs = model(inputs)
    
    # Get predicted class indices per pixel
    predictions = torch.argmax(outputs, dim=1)

if device.type == "cuda":
    torch.cuda.synchronize()
inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

# 6. Compute Quantitative Metrics
mean_dsc = compute_dsc(predictions, targets.squeeze(1).long())

print("\n--- Demonstration Results ---")
print(f"Inference Latency: {inference_time:.2f} ms for batch of size {inputs.size(0)}")
print(f"Test Batch Mean DSC: {mean_dsc:.4f}")

# 7. Generate Qualitative Visualization (First Slice in Batch)
img_input = inputs[0].cpu().squeeze().numpy()
img_target = targets[0].cpu().squeeze().numpy()
img_pred = predictions[0].cpu().squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_input, cmap="gray")
axes[0].set_title("Input MRI Slice", fontsize=12)

axes[1].imshow(img_target, cmap="tab20")
axes[1].set_title("Ground Truth Mask", fontsize=12)

axes[2].imshow(img_pred, cmap="tab20")
axes[2].set_title(f"UNet Prediction (DSC: {mean_dsc:.4f})", fontsize=12)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("demo_segmentation_result.png", dpi=300, bbox_inches="tight")
print("\nSaved qualitative plot to: demo_segmentation_result.png")