import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from PIL import Image

# 1. Setup & Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
latent_dim = 2
epochs = 30
lr = 1e-3

DATA_PATH = "/home/groups/comp3710/OASIS/keras_png_slices_train"

# Custom Dataset for flat image directories
class OASISDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # Convert to 1-channel Grayscale
        if self.transform:
            image = self.transform(image)
        return image, 0

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = OASISDataset(img_dir=DATA_PATH, transform=transform)
train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

# 2. VAE Model Architecture (128x128 Single-Channel Brain Slices)
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        # Encoder: 1x128x128 -> 256x8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        # Decoder: 2x1 -> 1x128x128
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.decoder_input(z)), mu, logvar

model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 3. Training Loop
print(f"Starting VAE Training on {device}...", flush=True)
model.train()
for epoch in range(1, epochs + 1):
    train_loss = 0
    for x, _ in train_loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f}", flush=True)

# 4. Generate & Save 2D Manifold Grid
print("Generating 2D Latent Manifold...", flush=True)
model.eval()
n = 15
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

fig, axes = plt.subplots(n, n, figsize=(10, 10))
with torch.no_grad():
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            recon = model.decoder(model.decoder_input(z)).cpu().squeeze().numpy()
            axes[i, j].imshow(recon, cmap="gray")
            axes[i, j].axis("off")

plt.tight_layout()
plt.savefig("vae_manifold.png", dpi=300)
print("Saved manifold visualization to vae_manifold.png", flush=True)