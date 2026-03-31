# definitiveDRIVE  
_Semantic Segmentation for Autonomous Driving_

---

## 📌 Project overview

**definitiveDRIVE** is an end-to-end application designed to demonstrate a complete semantic segmentation pipeline for autonomous driving use cases.

The project covers:
- image preprocessing and semantic segmentation with a DeepLabV3+ model,
- deployment of a prediction API,
- a web application allowing visual comparison between:
  - the real image,
  - the ground truth mask,
  - the predicted segmentation mask.

The application is built following a clean **API / Frontend separation**, as expected in real-world machine learning systems.

---

## 🎯 Objectives

- Demonstrate the feasibility of semantic segmentation for autonomous driving scenes.
- Serve predictions through a REST API.
- Provide a clear and interpretable visual interface for comparing predictions with ground truth.
- Use lightweight, deployment-ready tools.

---

## 🧠 Model

- Architecture: **DeepLabV3+**
- Backbone: **ResNet50**
- Input size: **256 × 512**
- Number of classes: **8**
- Training dataset: Cityscapes (validation samples reused for demonstration purposes)
- Output: pixel-wise semantic segmentation

---

## 🏗️ Project architecture

```
PROJET8_PROD
│
├── api/
│   ├── main.py                  # FastAPI application
│   ├── utils/
│   │   ├── inference.py         # Preprocessing & inference logic
│   │   └── data_loader.py       # Dataset access helpers
│   ├── models/
│   │   └── deeplabv3plus_model.keras
│   ├── data/
│   │   ├── images/              # Real images (PNG)
│   │   └── masks_real/          # Ground truth masks (PNG)
│   ├── requirements.txt
│   └── venv/
│
├── streamlit_app/
│   ├── app.py                   # Streamlit web application
│   ├── assets/
│   │   └── logo.png
│   └── requirements.txt
│
└── README.md
```

---

## 🌐 Cloud deployment

The application is fully deployed in the cloud using Render.

### 🔹 API (FastAPI)
👉 Swagger documentation available at:  
https://projet8-api.onrender.com/docs

### 🔹 Streamlit application
👉 Web interface available at:  
https://projet8-streamlit.onrender.com

### ⚠️ Important (demo preparation)
Render services may go idle after inactivity.  
For a smooth demonstration, open both URLs a few minutes before use to warm up the services.

---

## 🔌 API description (FastAPI)

The API exposes the following endpoints:

| Method | Endpoint | Description |
|------|---------|-------------|
| GET | `/images` | List available image IDs |
| GET | `/image/{image_id}` | Retrieve a real image |
| GET | `/mask-real/{image_id}` | Retrieve the ground truth mask |
| GET | `/predict/{image_id}` | Retrieve the predicted segmentation mask |

---

## 🖥️ Web application (Streamlit)

The Streamlit application allows users to:
1. Select a driving scene.
2. Run semantic segmentation.
3. Visualize side-by-side:
   - the real image,
   - the ground truth mask,
   - the predicted mask.

---

## 🚀 How to run the project locally

### 1️⃣  Start  the API

```bash
cd PROJET8_PROD
uvicorn api.main:app --reload
```
API available at : http://127.0.0.1:8000

### 2️⃣  Launch  the Streamlit application

```bash
cd PROJET8_PROD
streamlit run streamlit_app/app.py
```

Application available at : http://localhost:8501

---

## 🔒 Design choices

- Separation between API (inference) and frontend (visualization).
- Dynamic loading of the model from an external source (GitHub Release).
- Dual data loading strategy:
	•	local dataset for development,
	•	cloud fallback for deployment.
- Environment variable configuration (API_URL) for seamless local/cloud switching.
- Lightweight dataset (6 images) to enable cloud deployment without heavy storage constraints.

---

## 📦 Deployment strategy

- Docker-based deployment for reproducibility.
- API and frontend deployed as **independent services**.
- Cloud infrastructure handled via Render.
- Model and demo data loaded dynamically at runtime.

---

## 👩‍💻 Author

**Michèle Dewerpe**  
Project developed as part of the AI Engineer training program:   
"Traitez les images pour le système embarqué d’une voiture autonome".
