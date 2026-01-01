from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
import string
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
DATA_ROOT = "3_4_captcha/data"
OUTPUT_DIR = os.path.join(DATA_ROOT, "images")
CSV_FILE = os.path.join(DATA_ROOT, "dataset.csv")
NUM_IMAGES = 5000 # 5000 images simples
WIDTH = 200
HEIGHT = 64
ALPHABET = string.ascii_uppercase

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def generate_random_text(length=6):
    return ''.join(random.choices(ALPHABET, k=length))

def generate_image(text):
    image = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try: font = ImageFont.truetype("Arial.ttf", 36)
    except: font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (WIDTH - text_w) / 2
    y = (HEIGHT - text_h) / 2
    
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    # Bruit léger
    for _ in range(100):
        draw.point((random.randint(0, WIDTH), random.randint(0, HEIGHT)), fill=(100, 100, 100))
    return image

def generate_dataset():
    data = []
    print(f"Génération de {NUM_IMAGES} images SIMPLES...")
    for i in range(NUM_IMAGES):
        text = generate_random_text()
        filename = f"{i}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        generate_image(text).save(path)
        data.append([filename, text])
        if i % 1000 == 0: print(f"  {i}/{NUM_IMAGES}...")

    df = pd.DataFrame(data, columns=['filename', 'Label'])
    df.to_csv(CSV_FILE, index=False)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df.to_csv(os.path.join(DATA_ROOT, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_ROOT, "val.csv"), index=False)
    print("Terminé !")

if __name__ == "__main__":
    generate_dataset()
