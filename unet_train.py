import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_dataset import OASISDataset  # Ensure this matches your dataset filename
from unet import UNet

# --- Dice Loss Function ---
class CombinedLoss(nn.Module):
    def __init__(self, num_classes=65):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W], targets: [B, C, H, W] (one-hot)
        ce_loss = self.ce(inputs, targets.argmax(dim=1))
        
        probs = torch.softmax(inputs, dim=1)
        intersection = torch.sum(probs * targets, dim=(2, 3))
        cardinality = torch.sum(probs + targets, dim=(2, 3))
        
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return ce_loss + dice_loss.mean()

# --- Metric Calculation ---
def compute_dsc(preds, targets, num_classes=65):
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)
    
    dscs = []
    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets == c)
        intersection = (p_c & t_c).sum().float().item()
        total = p_c.sum().float().item() + t_c.sum().float().item()
        if total == 0:
            dscs.append(1.0)  # Class not present in ground truth or pred
        else:
            dscs.append((2.0 * intersection) / total)
    return dscs

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = OASISDataset(root_dir="/home/groups/comp3710/OASIS", split="train")
    val_ds = OASISDataset(root_dir="/home/groups/comp3710/OASIS", split="validate")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = UNet(in_channels=1, num_classes=65).to(device)
    criterion = CombinedLoss(num_classes=65)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation Step
        model.eval()
        val_dscs = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                dscs = compute_dsc(outputs, targets)
                val_dscs.append(dscs)

        avg_dsc = torch.tensor(val_dscs).mean().item()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f} - Mean Val DSC: {avg_dsc:.4f}")

    torch.save(model.state_dict(), "unet_oasis.pth")
    print("Model saved to unet_oasis.pth")

if __name__ == "__main__":
    train()