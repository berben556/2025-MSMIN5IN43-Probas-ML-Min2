import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import string

# Configuration
DATA_DIR = "../data/images"
TRAIN_CSV = "../data/train.csv"
IMG_WIDTH = 200
IMG_HEIGHT = 64 # Standard 64 ou 32
BATCH_SIZE = 32
ALPHABET = string.ascii_uppercase
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(ALPHABET)}
IDX2CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}

class CaptchaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self): return len(self.annotations)
    def __getitem__(self, index):
        img_name = self.annotations.iloc[index]['filename']
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert("L")
        except:
             image = Image.new('L', (IMG_WIDTH, IMG_HEIGHT))
        label_str = self.annotations.iloc[index]['Label']
        if self.transform: image = self.transform(image)
        label = torch.tensor([CHAR2IDX[c] for c in label_str if c in CHAR2IDX], dtype=torch.long)
        return image, label

# Architecture VGG-Style simple pour CRNN
class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        # Input: 1 x 64 x 200
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # -> 64 x 32 x 100
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # -> 128 x 16 x 50
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), # -> 256 x 16 x 50
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # -> 256 x 8 x 50
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), # -> 512 x 8 x 50
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # -> 512 x 4 x 50
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU() # -> 512 x 3 x 49
        )
        self.rnn = nn.LSTM(512 * 3, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.output = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        # [Batch, C, H, W] -> [Batch, W, C*H]
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        output, _ = self.rnn(features)
        return self.output(output).log_softmax(2)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long)
    targets = torch.cat(labels)
    return images, targets, target_lengths

def decode(preds):
    preds = preds.argmax(dim=2).detach().cpu().numpy()
    decoded = []
    for p in preds:
        text = ""
        for i in range(len(p)):
            char_idx = int(p[i])
            if char_idx != 0 and (i == 0 or char_idx != int(p[i-1])): text += IDX2CHAR.get(char_idx, "")
        decoded.append(text)
    return decoded

def train():
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = CaptchaDataset(csv_file=TRAIN_CSV, root_dir=DATA_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CRNN(num_chars=len(ALPHABET))
    device = torch.device("cpu")
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- CRASH TEST NOUVELLE ARCHI (100 iters sur 1 batch) ---")
    single_batch = next(iter(train_loader))
    img, tgt, tgt_len = single_batch
    
    for i in range(101):
        optimizer.zero_grad()
        preds = model(img)
        preds_p = preds.permute(1, 0, 2)
        input_len = torch.full(size=(img.size(0),), fill_value=preds.size(1), dtype=torch.long)
        loss = criterion(preds_p, tgt, input_len, tgt_len)
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            d = decode(preds)[0]
            print(f"Iter {i} | Loss: {loss.item():.4f} | Pred: {d}")

    # Si ça a marché, on sauvegarde pour que main.py puisse l'utiliser
    torch.save(model.state_dict(), "model.pth")
    print("Modèle sauvegardé (version crash test).")

if __name__ == "__main__":
    train()