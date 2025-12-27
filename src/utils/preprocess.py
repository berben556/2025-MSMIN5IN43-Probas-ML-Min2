import os
from PIL import Image

def preprocess_images(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_input = os.path.join(input_dir, class_name)
        class_output = os.path.join(output_dir, class_name)
        os.makedirs(class_output, exist_ok=True)

        for img_name in os.listdir(class_input):
            img_path = os.path.join(class_input, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            img.save(os.path.join(class_output, img_name))
