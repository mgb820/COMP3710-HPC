import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import model builder from main training script
from Demo_2_Part_3_CNN import get_cifar_resnet18

# 1. Device Setup & Class Definitions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Data Augmentation & Normalization
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

# 3. Load Datasets
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

# 4. Model & Optimizer Setup
model = get_cifar_resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# 5. Execute 1 Epoch of Training
print(f"--- Running Demo: 1 Training Epoch on {device} ---")
start_train_time = time.time()
model.train()
running_loss = 0.0

for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    running_loss += loss.item()

epoch_duration = time.time() - start_train_time
avg_loss = running_loss / len(train_loader)
print(f"Epoch 1 Complete | Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s\n")

# Save checkpoint after 1 epoch demo
torch.save(model.state_dict(), "demo_1epoch_model.pth")

# 6. Run Live Inference Sample
model.eval()
inputs, labels = next(iter(test_loader))
inputs, labels = inputs.to(device), labels.to(device)

start_infer_time = time.time()
with torch.no_grad():
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(inputs)
        _, predicted = outputs.max(1)

if device.type == "cuda":
    torch.cuda.synchronize()
latency_ms = (time.time() - start_infer_time) * 1000

# 7. Print Inference Results & Save Grid Plot
correct = predicted.eq(labels).sum().item()
accuracy = (correct / labels.size(0)) * 100

print("--- Live Inference Results ---")
print(f"Sample Batch Accuracy: {accuracy:.2f}% ({correct}/{labels.size(0)})")
print(f"Inference Latency:     {latency_ms:.2f} ms")

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

for i, ax in enumerate(axes.flat):
    img = inputs[i].cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    
    pred_label = classes[predicted[i]]
    true_label = classes[labels[i]]
    text_color = "green" if pred_label == true_label else "red"
    
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=text_color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("cifar10_demo_results.png", dpi=300)
print("Saved prediction output to 'cifar10_demo_results.png'")