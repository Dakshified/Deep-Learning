import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the PyTorch MLP model
class TitanicMLP(nn.Module):
    def __init__(self, input_dim):
        super(TitanicMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    print("="*60)
    print("Training Titanic Survival Classifier (PyTorch MLP)")
    print("="*60)

    # 1. Load Data
    csv_path = 'titanic (1).csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # 2. Preprocessing
    # Drop Name column
    df = df.drop(columns=['Name'])

    # Convert Sex to binary numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Separate features and target
    X = df.drop(columns=['Survived'])
    y = df['Survived'].values

    # Columns: Pclass, Sex, Age, Siblings/Spouses Aboard, Parents/Children Aboard, Fare
    feature_cols = X.columns.tolist()
    print("Features used:", feature_cols)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 3. Model Setup
    model = TitanicMLP(input_dim=X_train_scaled.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Training Loop
    epochs = 100
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Calculate train accuracy
        train_preds = (outputs >= 0.5).float()
        train_acc = (train_preds == y_train_tensor).float().mean().item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_preds = (test_outputs >= 0.5).float()
            test_acc = (test_preds == y_test_tensor).float().mean().item()

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(train_acc * 100)
        test_accs.append(test_acc * 100)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss.item():.4f} | Test Acc: {test_acc*100:.2f}%")

    # 5. Evaluate and Print Metrics
    model.eval()
    with torch.no_grad():
        final_test_outputs = model(X_test_tensor)
        y_pred = (final_test_outputs >= 0.5).int().numpy()
        
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

    # Save model weights
    torch.save(model.state_dict(), 'titanic_model.pth')
    print("Saved model weights to titanic_model.pth")

    # Save scale parameters for future use
    import pickle
    with open('titanic_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 6. Generate and Save Plots
    # Learning Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs+1), test_accs, label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('titanic_metrics.png')
    print("Saved metrics plot to titanic_metrics.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Titanic Prediction Confusion Matrix")
    plt.savefig('titanic_confusion_matrix.png')
    print("Saved confusion matrix to titanic_confusion_matrix.png")
    plt.close()

if __name__ == '__main__':
    main()
