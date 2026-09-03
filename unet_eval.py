import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from unet_dataset import OASISDataset
from unet import UNet

def compute_per_class_dsc(preds, targets, num_classes=65):
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)
    
    dsc_per_class = {}
    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets == c)
        intersection = (p_c & t_c).sum().float().item()
        total = p_c.sum().float().item() + t_c.sum().float().item()
        if total == 0:
            dsc_per_class[c] = 1.0
        else:
            dsc_per_class[c] = (2.0 * intersection) / total
    return dsc_per_class

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # Load Test Dataset
    test_ds = OASISDataset(root_dir="/home/groups/comp3710/OASIS", split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    # Load Model Weights
    model = UNet(in_channels=1, num_classes=65).to(device)
    model.load_state_dict(torch.load("unet_oasis.pth", map_location=device))
    model.eval()

    # Collect per-class DSC across test set
    all_class_dscs = {c: [] for c in range(65)}
    
    sample_img, sample_gt, sample_pred = None, None, None

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(test_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)

            dsc_dict = compute_per_class_dsc(outputs, targets)
            for c, dsc_val in dsc_dict.items():
                all_class_dscs[c].append(dsc_val)

            # Save the first sample for visualization
            if i == 0:
                sample_img = imgs[0, 0].cpu().numpy()
                sample_gt = torch.argmax(targets[0], dim=0).cpu().numpy()
                sample_pred = torch.argmax(outputs[0], dim=0).cpu().numpy()

    # Print Per-Class Summary
    print("\n--- Per-Class DSC Summary ---")
    failing_classes = []
    for c in range(65):
        mean_c_dsc = np.mean(all_class_dscs[c])
        if mean_c_dsc < 0.9:
            failing_classes.append((c, mean_c_dsc))
        print(f"Class {c:02d}: DSC = {mean_c_dsc:.4f}")

    if failing_classes:
        print(f"\n WARNING: {len(failing_classes)} classes did not reach >0.9 DSC target:")
        for c, score in failing_classes:
            print(f"  Class {c}: {score:.4f}")
    else:
        print("\n SUCCESS: All 65 classes achieved > 0.9 DSC target!")

    # Generate Comparative Plot for Report / Demo
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sample_img, cmap="gray")
    axes[0].set_title("Input MRI Slice")
    axes[0].axis("off")

    axes[1].imshow(sample_gt, cmap="tab20")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(sample_pred, cmap="tab20")
    axes[2].set_title("UNet Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_result.png", dpi=300)
    print("\nSaved visual proof plot to 'segmentation_result.png'")

if __name__ == "__main__":
    evaluate()