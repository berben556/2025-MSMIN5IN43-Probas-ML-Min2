import os
from PIL import Image
import numpy as np

# On charge une image générée
img_path = "3_4_captcha/data/images/0.png"

if os.path.exists(img_path):
    # Test 1: Chargement brut
    img = Image.open(img_path)
    print(f"Format original: {img.mode}")
    
    # Test 2: Conversion
    img_gray = img.convert("L")
    arr = np.array(img_gray)
    
    print(f"Stats pixel (Min/Max/Mean): {arr.min()}/{arr.max()}/{arr.mean():.2f}")
    
    if arr.mean() < 5 or arr.mean() > 250:
        print("ALERTE: L'image semble unie (tout noir ou tout blanc) !")
    else:
        print("L'image semble correcte (contient du contraste).")
else:
    print("Pas d'image trouvée.")
