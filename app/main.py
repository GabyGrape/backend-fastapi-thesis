import os
import cv2
import joblib
import glob
import time  # <--- Ditambahkan untuk mencatat durasi komputasi milidetik/detik
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from typing import List

# 1. Inisialisasi FastAPI
app = FastAPI(title="Chili Disease Detection API - Official Production Pipeline")

# 2. Pengaturan Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model, scaler, pca, le = None, None, None, None

# 3. Loading Models
try:
    model = joblib.load(os.path.join(BASE_DIR, "model_xgboost_cabai.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler_cabai.pkl"))
    pca = joblib.load(os.path.join(BASE_DIR, "pca_cabai.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "label_encoder_cabai.pkl"))
    print("✅ Pipeline ML Berhasil Dimuat (Stateless Mode)!")
except Exception as e:
    print(f"❌ Error Loading Models: {e}")

# --- FUNGSI HELPER: Ground Truth Logic ---
def get_ground_truth(file_name: str):
    fn = file_name.lower()
    if 'yellowish' in fn: return 'yellowish-leaf-geminivirus_chili-leaf-disease-dataset-main-ulfa-damayanti-dipakai-yellowish'
    if 'cercospora' in fn: return 'cercosporaleafspot_bintikdaun_dataset-skripsi_rahmat-kiswanto'
    if 'curl' in fn: return 'curlvirus_dataset-skripsi_rahmat-kiswanto'
    if 'healthy' in fn: return 'healthyleaf_dataset-skripsi_rahmat-kiswanto'
    return "unknown"

# 4. Fungsi Ekstraksi Fitur
def extract_features_from_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    # Preprocessing
    img = cv2.resize(img, (256, 256))
    img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    l, a, b_lab = cv2.split(lab)
    l2 = clahe.apply(l)
    img_final = cv2.cvtColor(cv2.merge((l2, a, b_lab)), cv2.COLOR_LAB2BGR)

    img_hsv = cv2.cvtColor(img_final, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
    
    # Warna
    h, s, v = cv2.split(img_hsv)
    features_color = [
        np.mean(h), np.std(h), skew(h.flatten()),
        np.mean(s), np.std(s), skew(s.flatten()),
        np.mean(v), np.std(v), skew(v.flatten())
    ]

    # Tekstur
    glcm = graycomatrix(img_gray, distances=[5], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    features_texture = [
        np.mean(graycoprops(glcm, prop)) for prop in 
        ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    ]

    # Bentuk
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    return np.array(features_color + features_texture + [edge_density])

# 5. Endpoints
@app.get("/")
def home():
    return {"status": "Active", "message": "Backend Skripsi Cabai Ready"}

# --- ENDPOINT 1: Prediksi Satu Gambar (Upload) ---
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if any(v is None for v in [model, scaler, pca, le]):
        raise HTTPException(status_code=500, detail="Model files missing.")

    # [PENGUKURAN] Catat waktu awal request diproses di server
    start_time = time.time()

    contents = await file.read()
    raw_features = extract_features_from_image(contents)
    
    if raw_features is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        feat_scaled = scaler.transform(raw_features.reshape(1, -1))
        feat_pca = pca.transform(feat_scaled)
        
        prediction = model.predict(feat_pca)
        proba = model.predict_proba(feat_pca)
        
        idx = int(prediction[0])
        label_full = str(le.classes_[idx])

        # [PENGUKURAN] Hitung durasi total komputasi server
        inference_time = time.time() - start_time

        return {
            "label": label_full, 
            "short_label": label_full.split('_')[0],
            "confidence": round(float(np.max(proba)), 4),
            "server_inference_time_sec": round(inference_time, 6),  # <--- Balikan data ke klien
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: Prediksi Batch untuk Mobile / Frontend ---
@app.post("/predict-images-batch")
async def predict_images_batch(files: List[UploadFile] = File(...)):
    if any(v is None for v in [model, scaler, pca, le]):
        raise HTTPException(status_code=500, detail="Model files missing.")

    if not files:
        return {"message": "Tidak ada file gambar yang dikirim.", "results": []}

    # [PENGUKURAN] Catat waktu awal proses batch dimulai
    start_batch_time = time.time()

    results = []
    correct_count = 0
    valid_validation_count = 0

    for file in files:
        try:
            file_name = file.filename
            contents = await file.read()
            
            raw_features = extract_features_from_image(contents)
            
            if raw_features is not None:
                feat_scaled = scaler.transform(raw_features.reshape(1, -1))
                feat_pca = pca.transform(feat_scaled)
                
                prediction = model.predict(feat_pca)
                proba = model.predict_proba(feat_pca)
                
                pred_label = str(le.classes_[int(prediction[0])])
                actual_label = get_ground_truth(file_name)
                
                status = "N/A"
                if actual_label != "unknown":
                    valid_validation_count += 1
                    if pred_label == actual_label:
                        status = "TRUE"
                        correct_count += 1
                    else:
                        status = "FALSE"

                results.append({
                    "file_name": file_name,
                    "actual": actual_label.split('_')[0] if actual_label != "unknown" else "unknown",
                    "prediction": pred_label.split('_')[0],
                    "confidence": round(float(np.max(proba)), 4),
                    "status": status
                })
            else:
                results.append({
                    "file_name": file_name,
                    "prediction": "Error",
                    "detail": "Gagal ekstraksi fitur dari gambar"
                })

        except Exception as e:
            results.append({
                "file_name": file.filename,
                "prediction": "Error",
                "detail": str(e)
            })

    # [PENGUKURAN] Hitung akumulasi durasi pemrosesan seluruh batch di server
    total_batch_server_time = time.time() - start_batch_time
    accuracy = (correct_count / valid_validation_count * 100) if valid_validation_count > 0 else 0

    return {
        "summary": {
            "total_files": len(results),
            "validated": valid_validation_count,
            "correct": correct_count,
            "accuracy_percent": round(accuracy, 2),
            "total_server_batch_time_sec": round(total_batch_server_time, 6)  # <--- Hasil ukur batch durasi server
        },
        "predictions": results
    }