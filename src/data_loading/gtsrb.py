"""gtsrb preprocessing used by the experiment models."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loaders(data_dir, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.GTSRB(data_dir, split='train', download=True, transform=transform)
    test_dataset  = datasets.GTSRB(data_dir, split='test',  download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, test_loader
