# import os
# import cv2
# import joblib
# import glob
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from scipy.stats import skew
# from skimage.feature import graycomatrix, graycoprops

# # 1. Inisialisasi FastAPI
# app = FastAPI(title="Chili Disease Detection API - Official Pipeline")

# # 2. Pengaturan Path
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model, scaler, pca, le = None, None, None, None

# # 3. Loading Models
# try:
#     model = joblib.load(os.path.join(BASE_DIR, "model_xgboost_cabai.pkl"))
#     scaler = joblib.load(os.path.join(BASE_DIR, "scaler_cabai.pkl"))
#     pca = joblib.load(os.path.join(BASE_DIR, "pca_cabai.pkl"))
#     le = joblib.load(os.path.join(BASE_DIR, "label_encoder_cabai.pkl"))
#     print("✅ Pipeline ML Berhasil Dimuat!")
# except Exception as e:
#     print(f"❌ Error Loading Models: {e}")

# # 4. Fungsi Ekstraksi Fitur (Harus sama persis dengan saat Training)
# def extract_features_from_image(image_bytes):
#     # Konversi bytes ke OpenCV Image
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None: 
#         return None

#     # --- PREPROCESSING ---
#     img = cv2.resize(img, (256, 256))
#     img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
    
#     # Perbaikan Kontras (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
#     l, a, b_lab = cv2.split(lab)
#     l2 = clahe.apply(l)
#     img_final = cv2.cvtColor(cv2.merge((l2, a, b_lab)), cv2.COLOR_LAB2BGR)

#     # Konversi Warna
#     img_hsv = cv2.cvtColor(img_final, cv2.COLOR_BGR2HSV)
#     img_gray = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
    
#     # --- A. Fitur Warna (9 Fitur: Mean, Std, Skew untuk H, S, V) ---
#     h, s, v = cv2.split(img_hsv)
#     features_color = [
#         np.mean(h), np.std(h), skew(h.flatten()),
#         np.mean(s), np.std(s), skew(s.flatten()),
#         np.mean(v), np.std(v), skew(v.flatten())
#     ]

#     # --- B. Fitur Tekstur GLCM (5 Fitur) ---
#     glcm = graycomatrix(img_gray, distances=[5], 
#                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
#                         levels=256, symmetric=True, normed=True)
    
#     features_texture = [
#         np.mean(graycoprops(glcm, prop)) for prop in 
#         ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
#     ]

#     # --- C. Fitur Bentuk (1 Fitur: Edge Density) ---
#     edges = cv2.Canny(img_gray, 100, 200)
#     edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

#     return np.array(features_color + features_texture + [edge_density])

# # 5. Endpoints
# @app.get("/")
# def home():
#     return {"status": "Active", "message": "Backend Skripsi Cabai Ready"}

# @app.post("/predict-image")
# async def predict_image(file: UploadFile = File(...)):
#     # Validasi keberadaan model
#     if any(v is None for v in [model, scaler, pca, le]):
#         raise HTTPException(status_code=500, detail="Model files missing on server.")

#     # Membaca konten file
#     contents = await file.read()
    
#     # Langkah krusial: Ekstraksi fitur dari gambar yang diupload
#     raw_features = extract_features_from_image(contents)
    
#     if raw_features is None:
#         raise HTTPException(status_code=400, detail="Format gambar tidak valid atau rusak.")

#     try:
#         # Pipeline Preprocessing: Reshape -> Scale -> PCA
#         feat_reshaped = raw_features.reshape(1, -1)
#         feat_scaled = scaler.transform(feat_reshaped)
#         feat_pca = pca.transform(feat_scaled)
        
#         # Prediksi menggunakan XGBoost
#         prediction = model.predict(feat_pca)
#         proba = model.predict_proba(feat_pca)
        
#         idx = int(prediction[0])
#         confidence = float(np.max(proba))

#         return {
#             "label": str(le.classes_[idx]), 
#             "confidence": round(confidence, 4),
#             "prediction_index": idx,
#             "status": "success"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")
    


# @app.post("/predict-folder")
# async def predict_folder(folder_path: str):
#     """
#     Endpoint untuk memprediksi seluruh gambar dalam satu folder lokal.
#     Input: Path folder (string) contoh: "C:/Dataset/Test_Cabai"
#     """
#     # 1. Validasi keberadaan folder
#     if not os.path.exists(folder_path):
#         raise HTTPException(status_code=404, detail="Folder tidak ditemukan.")

#     # 2. Ambil semua file gambar (jpg, jpeg, png)
#     extensions = ('*.jpg', '*.jpeg', '*.png')
#     image_files = []
#     for ext in extensions:
#         image_files.extend(glob.glob(os.path.join(folder_path, ext)))

#     if not image_files:
#         return {"message": "Tidak ada file gambar ditemukan di folder tersebut.", "results": []}

#     results = []

#     # 3. Looping setiap file untuk diprediksi
#     for file_path in image_files:
#         try:
#             with open(file_path, "rb") as f:
#                 img_bytes = f.read()
            
#             # Ekstraksi fitur
#             raw_features = extract_features_from_image(img_bytes)
            
#             if raw_features is not None:
#                 # Proses Pipeline
#                 feat_reshaped = raw_features.reshape(1, -1)
#                 feat_scaled = scaler.transform(feat_reshaped)
#                 feat_pca = pca.transform(feat_scaled)
                
#                 prediction = model.predict(feat_pca)
#                 proba = model.predict_proba(feat_pca)
                
#                 idx = int(prediction[0])
#                 confidence = float(np.max(proba))
                
#                 # Simpan hasil per file
#                 results.append({
#                     "file_name": os.path.basename(file_path),
#                     "prediction": str(le.classes_[idx]),
#                     "confidence_score": round(confidence, 4)
#                 })
#             else:
#                 results.append({
#                     "file_name": os.path.basename(file_path),
#                     "prediction": "Error",
#                     "detail": "Gagal ekstraksi fitur"
#                 })

#         except Exception as e:
#             results.append({
#                 "file_name": os.path.basename(file_path),
#                 "prediction": "Error",
#                 "detail": str(e)
#             })

#     return {
#         "total_files": len(results),
#         "folder_path": folder_path,
#         "predictions": results
#     }
import os
import cv2
import joblib
import glob
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

# 1. Inisialisasi FastAPI
app = FastAPI(title="Chili Disease Detection API - Official Pipeline")

# 2. Pengaturan Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model, scaler, pca, le = None, None, None, None

# 3. Loading Models
try:
    model = joblib.load(os.path.join(BASE_DIR, "model_xgboost_cabai.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler_cabai.pkl"))
    pca = joblib.load(os.path.join(BASE_DIR, "pca_cabai.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "label_encoder_cabai.pkl"))
    print("✅ Pipeline ML Berhasil Dimuat!")
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

        return {
            "label": label_full, 
            "short_label": label_full.split('_')[0],
            "confidence": round(float(np.max(proba)), 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: Prediksi Folder (Batch dengan Validasi) ---
@app.post("/predict-folder")
async def predict_folder(folder_path: str):
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder tidak ditemukan.")

    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        return {"message": "Tidak ada file gambar.", "results": []}

    results = []
    correct_count = 0
    valid_validation_count = 0

    for file_path in image_files:
        try:
            file_name = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            
            raw_features = extract_features_from_image(img_bytes)
            
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
        except Exception as e:
            results.append({"file_name": os.path.basename(file_path), "error": str(e)})

    accuracy = (correct_count / valid_validation_count * 100) if valid_validation_count > 0 else 0

    return {
        "summary": {
            "total_files": len(results),
            "validated": valid_validation_count,
            "correct": correct_count,
            "accuracy_percent": round(accuracy, 2)
        },
        "predictions": results
    }