"""gtsrb network and original training procedure."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.data_loading.gtsrb import get_loaders


class ArachneGTSRBModel(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        # Conv layer: 16 filters, kernel=3, padding=0 -> 46x46
        # MaxPool 2x2 -> 23x23
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        # FC input: 16 * 23 * 23 = 8464
        self.fc1 = nn.Linear(16 * 23 * 23, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # -> 16x23x23
        x = x.view(x.size(0), -1)                # -> 8464
        x = torch.relu(self.fc1(x))              # -> 512
        x = self.fc2(x)                          # -> 43
        return x


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def train_model(data_dir, save_dir, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(data_dir, batch_size=128)

    model = ArachneGTSRBModel(num_classes=43).to(device)
    print(f"Model: Conv(3->16) -> MaxPool -> FC(8464->512) -> FC(512->43)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            _, predicted = model(images).max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        test_acc  = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = save_dir / 'gtsrb_arachne_rq1_best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, save_path)
            print(f"  -> Best model saved ({best_acc:.2f}%)")

        scheduler.step()

    print("\n" + "=" * 60)
    print(f"Training Completed! Best test accuracy: {best_acc:.2f}%")
    print("=" * 60)
