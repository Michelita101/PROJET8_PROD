from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

# Dossiers data
IMAGES_DIR = BASE_DIR / "data" / "images"
MASKS_REAL_DIR = BASE_DIR / "data" / "masks_real"


def list_available_ids():
    """
    Retourne la liste des IDs disponibles à partir des images réelles.
    """
    return sorted([
        p.stem for p in IMAGES_DIR.glob("*.png")
    ])


def get_image_path(image_id: str) -> Path:
    """
    Chemin vers l'image réelle.
    """
    path = IMAGES_DIR / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_id}")
    return path


def get_real_mask_path(image_id: str) -> Path:
    """
    Chemin vers le masque réel.
    """
    path = MASKS_REAL_DIR / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Real mask not found: {image_id}")
    return path
