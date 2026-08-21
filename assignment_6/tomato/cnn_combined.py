import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# =====================================================================
# 1. Model Definitions
# =====================================================================

class CNN1Layer(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN1Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 128 * 128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNNPooling(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNPooling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 64 * 64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNNDense(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNDense, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 64 * 64, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

# =====================================================================
# 2. Training Helper Function
# =====================================================================

def train_and_evaluate(model, model_name, train_loader, val_loader, train_size, val_size, device, epochs=5):
    print("\n" + "="*50)
    print(f"Training Model: {model_name}")
    print("="*50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / train_size
        train_acc = correct_train / total_train * 100
        history['train_loss'].append(epoch_loss)

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val * 100
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
    return history

def main():
    torch.manual_seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        train_dir = os.path.join(base_dir, 'tomato', 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Could not find the dataset folder 'train' in {base_dir}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model1 = CNN1Layer(num_classes).to(device)
    model2 = CNNPooling(num_classes).to(device)
    model3 = CNNDense(num_classes).to(device)
    
    epochs = 5
    hist1 = train_and_evaluate(model1, "1-Layer CNN (Conv only)", train_loader, val_loader, train_size, val_size, device, epochs)
    hist2 = train_and_evaluate(model2, "CNN with Max Pooling", train_loader, val_loader, train_size, val_size, device, epochs)
    hist3 = train_and_evaluate(model3, "CNN with Max Pooling & Dense Hidden Layer", train_loader, val_loader, train_size, val_size, device, epochs)
    
    torch.save(model1.state_dict(), os.path.join(base_dir, 'combined_cnn_1_layer_weights.pth'))
    torch.save(model2.state_dict(), os.path.join(base_dir, 'combined_cnn_pooling_weights.pth'))
    torch.save(model3.state_dict(), os.path.join(base_dir, 'combined_cnn_dense_weights.pth'))

    # Plot comparisons
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), hist1['train_loss'], marker='o', label='1-Layer CNN')
    plt.plot(range(1, epochs+1), hist2['train_loss'], marker='s', label='CNN + Pooling')
    plt.plot(range(1, epochs+1), hist3['train_loss'], marker='^', label='CNN + Pooling + Dense')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), hist1['val_acc'], marker='o', label='1-Layer CNN')
    plt.plot(range(1, epochs+1), hist2['val_acc'], marker='s', label='CNN + Pooling')
    plt.plot(range(1, epochs+1), hist3['val_acc'], marker='^', label='CNN + Pooling + Dense')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'model_comparison.png'))
    plt.close()
    print("\nSaved comparative plot to: model_comparison.png")

if __name__ == '__main__':
    main()
