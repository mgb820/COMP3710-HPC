import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import model architecture from training file
from Demo_2_Part_3_CNN import get_cifar_resnet18

# 1. Setup Device & CIFAR-10 Class Names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Matching Normalization Transforms
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 3. Load Test Dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)

# 4. Load Model Architecture & Pre-trained Weights
model = get_cifar_resnet18().to(device)
model.load_state_dict(torch.load("cifar10_model.pth", map_location=device))
model.eval()

# 5. Run Live Inference
inputs, labels = next(iter(test_loader))
inputs, labels = inputs.to(device), labels.to(device)

start_time = time.time()
with torch.no_grad():
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(inputs)
        _, predicted = outputs.max(1)

if device.type == "cuda":
    torch.cuda.synchronize()
latency_ms = (time.time() - start_time) * 1000

# 6. Display Performance Metrics
correct = predicted.eq(labels).sum().item()
accuracy = (correct / labels.size(0)) * 100

print("--- CIFAR-10 ResNet-18 Live Demonstration ---")
print(f"Execution Device:  {device}")
print(f"Batch Latency:     {latency_ms:.2f} ms (16 images)")
print(f"Sample Accuracy:   {accuracy:.2f}% ({correct}/{labels.size(0)})")

# 7. Generate & Save Predictions Plot
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# Mean and std tensors for un-normalizing images back to [0, 1] range for plotting
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
print("Saved prediction grid to 'cifar10_demo_results.png'")