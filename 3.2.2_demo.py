import time
import torch
import torchvision
import torchvision.transforms as transforms
from Demo_2_Part_3_CNN import get_cifar_resnet18
import matplotlib.pyplot as plt

# 1. Setup Device & Transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 2. Load CIFAR-10 Test Dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

# 3. Load Saved Weights
model = get_cifar_resnet18().to(device)
model.load_state_dict(torch.load("cifar10_model.pth", map_location=device))
model.eval()

# 4. Run Live Inference
inputs, labels = next(iter(testloader))
inputs, labels = inputs.to(device), labels.to(device)

start_time = time.time()
with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

if device.type == "cuda":
    torch.cuda.synchronize()
inference_time = (time.time() - start_time) * 1000

# 5. Print Performance
correct = (predicted == labels).sum().item()
accuracy = (correct / labels.size(0)) * 100

print(f"--- CIFAR-10 Live Demonstration ---")
print(f"Running on Device: {device}")
print(f"Inference Latency: {inference_time:.2f} ms for batch of 16")
print(f"Batch Accuracy: {accuracy:.2f}% ({correct}/{labels.size(0)})")

# 6. Save Plot Comparison
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = inputs[i].cpu() / 2 + 0.5  # Unnormalize
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("cifar10_demo_results.png", dpi=300)
print("Saved comparison image to cifar10_demo_results.png")