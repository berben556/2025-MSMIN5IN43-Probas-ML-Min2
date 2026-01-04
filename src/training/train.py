import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import sys
sys.path.append(".")

from src.datasets.mri_dataset import MRIDataset
from src.models.resnet import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = MRIDataset("data/processed/train", transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = get_model(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "experiments/model.pth")
print("Modèle sauvegardé")
