"""
API de segmentation sémantique – Projet Voiture Autonome (OpenClassrooms)

Endpoints principaux :
- GET /images
- GET /image/{image_id}
- GET /mask-real/{image_id}
- GET /predict/{image_id}

Endpoints techniques (optionnels) :
- POST /predict
- POST /predict-image-file
- POST /predict-image
"""

import io
import base64
from pathlib import Path

from PIL import Image
import tensorflow as tf
import keras
from keras.models import load_model
from keras.saving import register_keras_serializable

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from api.utils.inference import predict_mask, colorize_mask
from api.utils.data_loader import (list_available_ids, get_image_path, get_real_mask_path)


# Fonction nécessaire à la désérialisation du modèle (Lambda ASPP)
@register_keras_serializable()
def resize_to_16x32(t):
    return tf.image.resize(t, (16, 32), method='bilinear')

# Autoriser explicitement la désérialisation non-safe des lambda
keras.config.enable_unsafe_deserialization()

# Chargement du modèle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "deeplabv3plus_model.keras"
print(f"Chargement du modèle depuis : {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)
print("Modèle chargé avec succès ;-)")

app = FastAPI(
    title="DeepLabV3+ Segmentation API",
    description="API de segmentation sémantique (projet voiture autonome - OC)",
    version="1.0.0"
)

# HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "ok"}

# Liste des images disponibles
@app.get("/images")
def get_images():
    """
    Liste les IDs des images disponibles.
    """
    return {"images": list_available_ids()}

# Images réélles
@app.get("/image/{image_id}")
def get_real_image(image_id: str):
    """
    Retourne l'image réelle (leftImg8bit).
    """
    image_path = get_image_path(image_id)
    return FileResponse(image_path, media_type="image/png")

# Masques réels
@app.get("/mask-real/{image_id}")
def get_real_mask(image_id: str):
    """
    Retourne le masque réel (gtFine_color).
    """
    mask_path = get_real_mask_path(image_id)
    return FileResponse(mask_path, media_type="image/png")

# Comparaison réel vs prédit
@app.get("/predict/{image_id}")
def predict_from_id(image_id: str):
    """
    Prédit le masque à partir d'une image réelle identifiée par son ID.
    Retourne le masque colorisé en PNG.
    """
    # 1. Charger l'image réelle
    image_path = get_image_path(image_id)
    image = Image.open(image_path).convert("RGB")

    # 2. Passage en buffer mémoire (comme pour l'upload)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # 3. Prédiction
    mask = predict_mask(buf, model)

    # 4. Colorisation
    colorized = colorize_mask(mask)

    # 5. Retour en PNG
    output_buffer = io.BytesIO()
    colorized.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="image/png"
    )

# PREDICTION JSON
@app.post("/predict")
async def predict_json(file: UploadFile = File(...)):
    """
    Retourne le masque sous forme de liste JSON (H x W)
    """
    # Lecture du fichier
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Sauvegarde temporaire en mémoire
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # Lance la prédiction
    mask = predict_mask(buf, model)

    # Conversion en liste Python pour JSON
    mask_list = mask.tolist()

    return {"mask": mask_list}


# PREDICTION IMAGE COLORISEE EN PNG
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Retourne un PNG colorisé encodé en Base64
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Buffer mémoire pour l'inférence
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # Prédiction
    mask = predict_mask(buf, model)

    # Colorisation
    colorized = colorize_mask(mask)

    # Conversion en PNG Base64
    output_buffer = io.BytesIO()
    colorized.save(output_buffer, format="PNG")
    encoded_png = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

    return JSONResponse(content={"image_base64": encoded_png})

@app.post("/predict-image-file")
async def predict_image_file(file: UploadFile = File(...)):
    """
    Retourne l'image PNG colorisée comme un vrai fichier téléchargeable.
    """
    contents = await file.read()

    # Lecture de l’image envoyée
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Passage en buffer mémoire
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # Prédiction
    mask = predict_mask(buf, model)

    # Colorisation du mask
    colorized = colorize_mask(mask)

    # Conversion en PNG dans un buffer
    output_buffer = io.BytesIO()
    colorized.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    # Response fichier (téléchargeable dans Swagger)
    return StreamingResponse(
        output_buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": 'attachment; filename="segmentation.png"'
        }
    )
