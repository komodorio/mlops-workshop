import os
import tempfile

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from ray import tune
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-tracking")
mlflow.set_experiment("finger-counting")


class FingerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Assume structure: data_dir/0/, data_dir/1/, ..., data_dir/5/
        for label in range(6):  # 0-5 fingers
            label_dir = os.path.join(data_dir, str(label))
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.images.append(os.path.join(label_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6, width=32, depth=3):
        super(SimpleCNN, self).__init__()
        self.depth = depth
        self.width = width
        self.num_classes = num_classes

        # Build conv layers dynamically
        self.conv_layers = nn.ModuleList()
        in_channels = 3

        for i in range(depth):
            out_channels = width * (2**i)  # width, width*2, width*4, ...
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(2, 2)

        # Calculate spatial dimensions after pooling
        # Input: 224x224, after depth pooling operations: 224 / (2^depth)
        spatial_dim = 224 // (2**depth)
        fc_input_size = in_channels * spatial_dim * spatial_dim

        self.fc1 = nn.Linear(fc_input_size, width * 8)  # Proportional to width
        self.fc2 = nn.Linear(width * 8, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.print_config()

    def forward(self, x):
        # Apply conv layers with pooling
        for conv_layer in self.conv_layers:
            x = self.pool(torch.relu(conv_layer(x)))

        # Flatten and apply FC layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def print_config(self):
        """Print SimpleCNN configuration"""
        print(f"SimpleCNN Configuration:")
        print(f"  Depth: {self.depth} conv layers")
        print(f"  Base width: {self.width}")
        print(f"  Channel progression: 3 -> ", end="")
        for i in range(self.depth):
            channels = self.width * (2**i)
            print(f"{channels}", end=" -> " if i < self.depth - 1 else "\n")
        print(f"  Spatial reduction: 224x224 -> {224 // (2 ** self.depth)}x{224 // (2 ** self.depth)}")
        print(f"  FC layers: {self.fc1.in_features} -> {self.fc1.out_features} -> {self.fc2.out_features}")
        print()


def get_model(model_type, num_classes=6):
    if model_type == "cnn_deep":
        return SimpleCNN(num_classes, 4, 7)
    elif model_type == "cnn_wide":
        return SimpleCNN(num_classes, 16, 3)
    elif model_type == "mobilenet_v2":
        model = models.mobilenet_v2()
        # Freeze early layers
        for param in model.features[:-3].parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif model_type == "resnet18":
        model = models.resnet18()
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(config, data_root="/data"):
    # Set up data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    print("Loading datasets")
    train_dataset = FingerDataset(data_dir=f"{data_root}/{config['dataset_size']}/train", transform=transform)
    val_dataset = FingerDataset(data_dir=f"{data_root}/{config['dataset_size']}/val", transform=transform)
    print("Loaded datasets: %s train, %s val" % (len(train_dataset), len(val_dataset)))

    assert len(train_dataset), "Empty training dataset!"
    assert len(val_dataset), "Empty training dataset!"

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = get_model(config["model_type"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))

    # Training loop
    epochs = config.get("epochs", 10)
    device = torch.device("cpu")  # CPU only for workshop
    model.to(device)

    with mlflow.start_run(run_name=f"{config['model_type']}_{config['dataset_size']}"):
        mlflow.log_params(config)

        for epoch in range(epochs):
            print("Epoch #%s" % epoch)

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            train_accuracy = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

            val_accuracy = val_correct / val_total

            # Log metrics
            mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Epoch {epoch + 1}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
        )

        # Report final metrics to Ray Tune
        tune.report(
            metrics={
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": train_accuracy,
                "val_loss": val_loss / len(val_loader),
                "val_accuracy": val_accuracy,
            }
        )


if __name__ == "__main__":
    mlflow.set_experiment("finger-counting-debug")

    # For standalone testing - fast demo config
    config = {
        "model_type": "cnn_deep",
        "dataset_size": "small",
        "lr": 0.001,
        "epochs": 10,
    }
    train_model(config)
