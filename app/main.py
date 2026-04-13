# import os
# import cv2
# import joblib
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from scipy.stats import skew
# from skimage.feature import graycomatrix, graycoprops


# # 1. Inisialisasi FastAPI (WAJIB ADA AGAR TIDAK ERROR)
# app = FastAPI(title="Chili Disease Detection API - Official Pipeline")

# # 2. Pengaturan Path dan Variabel Global
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model, scaler, pca = None, None, None

# # Tambahkan inisialisasi CLAHE di bagian atas (di bawah imports)
# IMG_SIZE = (256, 256)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# class_names = [
#     "Cercospora Leaf Spot", 
#     "Curl Virus", 
#     "Healthy Leaf", 
#     "Yellowish Leaf"
# ]

# # 3. Loading Models
# try:
#     model = joblib.load(os.path.join(BASE_DIR, "model_xgboost_cabai.pkl"))
#     scaler = joblib.load(os.path.join(BASE_DIR, "scaler_cabai.pkl"))
#     pca = joblib.load(os.path.join(BASE_DIR, "pca_cabai.pkl"))
#     print("✅ Pipeline ML Berhasil Dimuat!")
# except Exception as e:
#     print(f"❌ Error Loading Models: {e}")

# # 4. Fungsi Ekstraksi 15 Fitur (IDENTIK DENGAN COLAB)
# def extract_features_from_image(image_bytes):
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None: return None

#     # --- LANGKAH 1: PREPROCESSING (IDENTIK DENGAN COLAB) ---
#     # 1. Resize
#     img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
#     # 2. Denoising (Gaussian Filter)
#     img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
#     # 3. CLAHE Enhancement
#     lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     l2 = clahe.apply(l)
#     lab = cv2.merge((l2, a, b))
#     img_final = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#     # --- LANGKAH 2: EKSTRAKSI FITUR ---
#     img_hsv = cv2.cvtColor(img_final, cv2.COLOR_BGR2HSV)
#     img_gray = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
    
#     # A. Warna (HSV) - 9 Fitur
#     h, s, v = cv2.split(img_hsv)
#     features_color = [
#         np.mean(h), np.std(h), skew(h.flatten()),
#         np.mean(s), np.std(s), skew(s.flatten()),
#         np.mean(v), np.std(v), skew(v.flatten())
#     ]

#     # B. Tekstur (GLCM) - 5 Fitur (Update distances=5 dan 4 angles)
#     glcm = graycomatrix(img_gray, distances=[5], 
#                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
#                         levels=256, symmetric=True, normed=True)
    
#     props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
#     features_texture = [np.mean(graycoprops(glcm, prop)) for prop in props]

#     # C. Bentuk (Edge Density) - 1 Fitur
#     edges = cv2.Canny(img_gray, 100, 200)
#     edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

#     return np.array(features_color + features_texture + [edge_density])
    
# # 5. Endpoints
# @app.get("/")
# def home():
#     return {"status": "Active", "message": "Backend Skripsi Cabai Ready"}

# @app.post("/predict-image")
# async def predict_image(file: UploadFile = File(...)):
#     if any(v is None for v in [model, scaler, pca]):
#         raise HTTPException(status_code=500, detail="Model files missing on server.")

#     contents = await file.read()
#     raw_features = extract_features_from_image(contents)
    
#     if raw_features is None:
#         raise HTTPException(status_code=400, detail="Invalid image file.")

#     try:
#         # Preprocessing: Reshape -> Scale -> PCA
#         feat_reshaped = raw_features.reshape(1, -1)
#         feat_scaled = scaler.transform(feat_reshaped)
#         feat_pca = pca.transform(feat_scaled)
        
#         # Prediksi
#         prediction = model.predict(feat_pca)
#         proba = model.predict_proba(feat_pca)
        
#         idx = int(prediction[0])
#         confidence = float(np.max(proba))

#         return {
#             "label": class_names[idx],
#             "confidence": round(confidence, 4),
#             "prediction_index": idx
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
import os
import cv2
import joblib
import numpy as np
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from motor.motor_asyncio import AsyncIOMotorClient

# 1. Inisialisasi FastAPI
app = FastAPI(title="Chili Disease Detection API - Production Pipeline")

# 2. Konfigurasi MongoDB
# Pastikan MongoDB Service sudah running di laptop MSI kamu
MONGO_DETAILS = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.bercabai_db  # Nama Database
collection = database.prediction_history  # Nama Koleksi (Tabel)

# 3. Pengaturan Path dan Variabel Global
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model, scaler, pca = None, None, None

IMG_SIZE = (256, 256)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

class_names = [
    "Cercospora Leaf Spot", 
    "Curl Virus", 
    "Healthy Leaf", 
    "Yellowish Leaf"
]

# 4. Loading Models
try:
    model = joblib.load(os.path.join(BASE_DIR, "model_xgboost_cabai.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler_cabai.pkl"))
    pca = joblib.load(os.path.join(BASE_DIR, "pca_cabai.pkl"))
    print("✅ Pipeline ML Berhasil Dimuat!")
except Exception as e:
    print(f"❌ Error Loading Models: {e}")

# 5. Fungsi Ekstraksi 15 Fitur (Identik dengan Colab)
def extract_features_from_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    # --- PREPROCESSING ---
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img_final = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- EKSTRAKSI FITUR ---
    img_hsv = cv2.cvtColor(img_final, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
    
    # A. Warna (HSV)
    h, s, v = cv2.split(img_hsv)
    features_color = [
        np.mean(h), np.std(h), skew(h.flatten()),
        np.mean(s), np.std(s), skew(s.flatten()),
        np.mean(v), np.std(v), skew(v.flatten())
    ]

    # B. Tekstur (GLCM)
    glcm = graycomatrix(img_gray, distances=[5], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features_texture = [np.mean(graycoprops(glcm, prop)) for prop in props]

    # C. Bentuk (Edge Density)
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    return np.array(features_color + features_texture + [edge_density])

# 6. Endpoints
@app.get("/")
def home():
    return {"status": "Active", "message": "Backend Skripsi Cabai Ready"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if any(v is None for v in [model, scaler, pca]):
        raise HTTPException(status_code=500, detail="Model files missing on server.")

    contents = await file.read()
    raw_features = extract_features_from_image(contents)
    
    if raw_features is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        # Preprocessing Inference: Reshape -> Scale -> PCA
        feat_reshaped = raw_features.reshape(1, -1)
        feat_scaled = scaler.transform(feat_reshaped)
        feat_pca = pca.transform(feat_scaled)
        
        # Prediksi
        prediction = model.predict(feat_pca)
        proba = model.predict_proba(feat_pca)
        
        idx = int(prediction[0])
        label_name = class_names[idx]
        confidence = float(np.max(proba))

        # --- LOGIKA MONGODB: Simpan Riwayat ---
        history_data = {
            "filename": file.filename,
            "prediction": label_name,
            "confidence": round(confidence, 4),
            "timestamp": datetime.datetime.now(), # Waktu otomatis
            "model_type": "XGBoost + PCA"
        }
        await collection.insert_one(history_data)

        # Response ke Flutter
        return {
            "status": "success",
            "label": label_name,
            "confidence": round(confidence, 4),
            "prediction_index": idx
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))