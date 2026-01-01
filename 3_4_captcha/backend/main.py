from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import pandas as pd
import random
from captcha.image import ImageCaptcha 

# Architecture VGG-Style (Copie de train_model.py)
class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        # Input: 1 x 64 x 200
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # -> 64 x 32 x 100
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # -> 128 x 16 x 50
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), 
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # -> 256 x 8 x 50
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # -> 512 x 4 x 50
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU() # -> 512 x 3 x 49
        )
        self.rnn = nn.LSTM(512 * 3, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.output = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        output, _ = self.rnn(features)
        return self.output(output).log_softmax(2)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-True-Label", "X-Prediction"],
)

MODEL_PATH = "model.pth"
DATA_DIR = "../data/images" 
VAL_CSV = "../data/val.csv"
IMG_WIDTH = 200
IMG_HEIGHT = 64 # UPDATED
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX2CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}

device = torch.device("cpu")
model = CRNN(num_chars=len(ALPHABET))
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Modèle chargé.")
    except:
        print("Erreur chargement modèle (architecture incompatible ?)")
else:
    print("Modèle non trouvé.")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode_prediction(preds):
    preds = preds.argmax(dim=2).detach().cpu().numpy()
    decoded_texts = []
    for p in preds:
        text = ""
        for i in range(len(p)):
            char_idx = int(p[i])
            if char_idx != 0 and (i == 0 or char_idx != int(p[i-1])):
                text += IDX2CHAR.get(char_idx, "")
        decoded_texts.append(text)
    return decoded_texts

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("L")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    text = decode_prediction(output)[0]
    return {"prediction": text}

@app.get("/test-sample")
def get_test_sample():
    if not os.path.exists(VAL_CSV):
        return {"error": "Validation set not found"}
    df = pd.read_csv(VAL_CSV)
    if len(df) == 0:
        return {"error": "Validation set empty"}
    random_row = df.sample(1).iloc[0]
    img_name = random_row['filename']
    true_label = random_row['Label']
    img_path = os.path.join(DATA_DIR, img_name)
    if not os.path.exists(img_path):
        return {"error": f"Image {img_name} not found"}
    return FileResponse(img_path, headers={"X-True-Label": true_label})

@app.post("/generate-custom")
async def generate_custom(text: str = Form(...)):
    text = text.upper()
    text = ''.join([c for c in text if c in ALPHABET])
    if not text: return {"error": "Invalid text"}

    image_generator = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    data = image_generator.generate(text)
    image = Image.open(data)
    
    img_gray = image.convert("L")
    img_tensor = transform(img_gray).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    prediction = decode_prediction(output)[0]
    
    data.seek(0)
    return StreamingResponse(data, media_type="image/png", headers={
        "X-True-Label": text,
        "X-Prediction": prediction
    })

@app.get("/")
def read_root():
    return {"message": "Captcha Solver API Ready (VGG Style)"}
