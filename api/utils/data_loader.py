from pathlib import Path
import urllib.request
import zipfile

# URL du zip de données de démonstration
DATA_URL = "https://github.com/Michelita101/PROJET8_PROD/releases/download/v1-data/demo_data.zip"

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

# Dossiers data locaux
IMAGES_DIR = BASE_DIR / "data" / "images"
MASKS_REAL_DIR = BASE_DIR / "data" / "masks_real"

# Dossiers cloud temporaires
TMP_ROOT_DIR = Path("/tmp/data")
TMP_EXTRACTED_DATA_DIR = TMP_ROOT_DIR / "data"
TMP_IMAGES_DIR = TMP_EXTRACTED_DATA_DIR / "images"
TMP_MASKS_REAL_DIR = TMP_EXTRACTED_DATA_DIR / "masks_real"

def _ensure_cloud_data():
    """
    Télécharge et extrait les données de démonstration si elles ne sont pas déjà disponibles.
    """
    if TMP_IMAGES_DIR.exists() and TMP_MASKS_REAL_DIR.exists():
        return

    TMP_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TMP_ROOT_DIR / "demo_data.zip"

    if not zip_path.exists():
        print(f"Téléchargement des données depuis : {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        print("Téléchargement des données terminé.")

    print(f"Extraction des données dans : {TMP_ROOT_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TMP_ROOT_DIR)
    print("Extraction terminée.")

def _get_data_dirs():
    """
    Retourne les bons dossiers selon l'environnement :
    - local si les dossiers natifs existent
    - cloud sinon, après téléchargement du zip
    """
    if IMAGES_DIR.exists() and MASKS_REAL_DIR.exists():
        return IMAGES_DIR, MASKS_REAL_DIR

    _ensure_cloud_data()
    return TMP_IMAGES_DIR, TMP_MASKS_REAL_DIR

def list_available_ids():
    """
    Retourne la liste des IDs disponibles à partir des images réelles.
    """
    images_dir, _ = _get_data_dirs()
    return sorted([p.stem for p in images_dir.glob("*.png")])

def get_image_path(image_id: str) -> Path:
    """
    Chemin vers l'image réelle.
    """
    images_dir, _ = _get_data_dirs()
    path = images_dir / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_id}")
    return path


def get_real_mask_path(image_id: str) -> Path:
    """
    Chemin vers le masque réel.
    """
    _, masks_real_dir = _get_data_dirs()
    path = masks_real_dir / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Real mask not found: {image_id}")
    return path
