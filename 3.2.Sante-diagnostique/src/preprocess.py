from src.utils.preprocess import preprocess_images

preprocess_images(
    "data/raw/Training",
    "data/processed/train"
)

preprocess_images(
    "data/raw/Testing",
    "data/processed/test"
)

print("Preprocessing terminé")
