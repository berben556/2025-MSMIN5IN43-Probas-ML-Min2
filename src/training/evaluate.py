import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.datasets.mri_dataset import MRIDataset
from src.models.resnet import get_model

transform = transforms.ToTensor()
test_dataset = MRIDataset("data/processed/test", transform)
test_loader = DataLoader(test_dataset, batch_size=16)

model = get_model(4)
model.load_state_dict(torch.load("experiments/model.pth"))
model.eval()

correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {correct / total:.2f}")
