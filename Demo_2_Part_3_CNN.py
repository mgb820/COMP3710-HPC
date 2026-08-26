import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 2. CIFAR-10 Optimized Data Augmentation
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
        transforms.RandomErasing(p=0.5, value="random"),  # Cutout regularizer
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

# Load Datasets
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)


# 3. ResNet-18 Modified Stem for 32x32 CIFAR-10 Images
def get_cifar_resnet18():
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    # Replace standard 7x7 conv (stride 2) with 3x3 conv (stride 1) to retain spatial size
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # Remove initial maxpool to prevent aggressive downsampling
    model.maxpool = nn.Identity()
    return model


model = get_cifar_resnet18().to(device)

# 4. Training Hyperparameters & AMP Scaler
epochs = 24
max_lr = 0.1
weight_decay = 5e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=max_lr, momentum=0.9, weight_decay=weight_decay
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.2,
)

# PyTorch Mixed Precision Scaler
scaler = torch.amp.GradScaler("cuda")

# 5. Training Loop
start_time = time.time()
print(f"Starting DAWNBench Fast-CIFAR10 Training on {device}...\n")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        running_loss += loss.item()

    # Intermediate status
    if (epoch + 1) % 6 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {running_loss/len(train_loader):.4f}"
        )

total_training_time = time.time() - start_time
print(f"\nTotal Training Duration: {total_training_time:.2f} seconds")

# 6. Evaluation Loop (Inference Benchmark)
model.eval()
correct = 0
total = 0
inference_start = time.time()

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

inference_time = time.time() - inference_start
accuracy = 100.0 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Inference Time for 10k Images: {inference_time:.4f} seconds")