import numpy as np
from typing import Tuple
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# --- Paramètres fixes du modèle ---
INPUT_HEIGHT = 256   # à adapter
INPUT_WIDTH = 512    # idem
NUM_CLASSES = 8        # 8 classes principales Cityscapes


# Palette de couleurs simple
PALETTE_8CLASS = np.array([
    [128, 64, 128],   # 0: route
    [244, 35, 232],   # 1: trottoir
    [70, 70, 70],     # 2: bâtiments/murs
    [220, 220, 0],    # 3: panneaux/feux
    [107, 142, 35],   # 4: végétation/terrain
    [70, 130, 180],   # 5: ciel
    [220, 20, 60],    # 6: piétons/riders
    [0, 0, 142],      # 7: véhicules
])

def preprocess_image_inference(image_buffer):
    """
    Prétraitement d'inférence aligné sur l'entraînement :
    - resize direct en (256, 512)
    - normalisation ResNet50
    """
    img = Image.open(image_buffer).convert("RGB")
    img = img.resize((512, 256), Image.BILINEAR)  # (W, H)

    img_np = np.array(img).astype(np.float32)

    # Normalisation ResNet50
    img_np = preprocess_input(img_np)

    x = np.expand_dims(img_np, axis=0)  # (1, 256, 512, 3)

    return x

def predict_mask(image_buffer, model):
    """
    Prédit le mask depuis un buffer image + modèle déjà chargé.
    """
    x = preprocess_image_inference(image_buffer)
    y_pred = model.predict(x)  # (1, H, W, NUM_CLASSES)
    mask = np.argmax(y_pred, axis=-1)[0]  # (H, W)
    return mask

def colorize_mask(mask: np.ndarray) -> Image.Image:
    """
    Transforme un mask de classes (H, W) en image RGB colorée selon PALETTE_8CLASS.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = PALETTE_8CLASS[class_id]

    return Image.fromarray(color_mask)


