import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Setup & Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# 2. Model Setup
def get_cifar_resnet18():
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

model = get_cifar_resnet18().to(device)

# UNCOMMENT TO LOAD PRE-TRAINED WEIGHTS FOR HIGH ACCURACY DEMO:
# model.load_state_dict(torch.load("resnet18_cifar.pth", map_location=device))

# 3. Demo Settings (Single Epoch)
epochs = 1
max_lr = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs
)
scaler = torch.amp.GradScaler("cuda")

# 4. Live Demonstration: Single Training Epoch
print(f"\n--- Starting Demonstration Epoch on {device} ---")
model.train()
start_train = time.time()
running_loss = 0.0

for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    running_loss += loss.item()

    # Print live progress every 100 batches during demo
    if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
        print(f"Batch [{batch_idx+1:03d}/{len(train_loader)}] | Current Loss: {loss.item():.4f}")

train_time = time.time() - start_train
print(f"Epoch Complete in {train_time:.2f} seconds | Average Loss: {running_loss/len(train_loader):.4f}")

# 5. Live Demonstration: Inference Pass
print("\n--- Running Inference Benchmark ---")
model.eval()
correct, total = 0, 0
start_infer = time.time()

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

infer_time = time.time() - start_infer
acc = 100.0 * correct / total

print(f"Inference Time (10k images): {infer_time:.3f} seconds")
print(f"Test Accuracy: {acc:.2f}%\n")