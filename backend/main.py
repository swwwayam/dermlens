"""
DermLens FastAPI Backend
========================
Asynchronous FastAPI server wrapping the HAM10000 CBIR inference pipeline.

- Pre-trained ResNet-50 (2048-D unit feature embeddings)
- FAISS IndexFlatIP (cosine similarity)
- Multimodal clinical metadata fusion (Age delta, Sex match, Anatomical localization)
- Similarity-gated rarity re-ranking (locked parameters from src/final_combined_test.py)
"""

import os
import sys
import time
import uuid
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import faiss
import torch
from torchvision import models, transforms

# ── Paths ─────────────────────────────────────────────────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FAISS_DIR   = os.path.join(BASE_DIR, "faiss_index")

EMBEDDINGS_PATH = os.path.join(RESULTS_DIR, "train_embeddings.npy")
METADATA_PATH   = os.path.join(RESULTS_DIR, "train_embedding_metadata.csv")
INDEX_PATH      = os.path.join(FAISS_DIR,   "ham10000_resnet.index")
FULL_META_PATH  = os.path.join(DATA_DIR,    "HAM10000_metadata.csv")

IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]

UPLOAD_DIR = os.path.join(FRONTEND_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Diagnostic taxonomy ───────────────────────────────────────
DX_LABELS = {
    "nv":    "Melanocytic Nevi",
    "mel":   "Melanoma",
    "bkl":   "Benign Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses",
    "vasc":  "Vascular Lesions",
    "df":    "Dermatofibroma",
}

DX_CATEGORIES = {
    "nv":    "Benign",
    "bkl":   "Benign",
    "df":    "Benign",
    "vasc":  "Benign",
    "mel":   "Malignant",
    "bcc":   "Malignant",
    "akiec": "Pre-malignant",
}

# ── Locked scoring constants (src/final_combined_test.py) ─────
CANDIDATE_K          = 50
SIMILARITY_THRESHOLD = 0.80
RARITY_BOOST         = 0.03
AGE_WEIGHT           = 0.03
SEX_WEIGHT           = 0.01
LOCATION_WEIGHT      = 0.03

# ── Device & Model initialization ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DermLens FastAPI] Execution Device: {device}")
if torch.cuda.is_available():
    print(f"[DermLens FastAPI] GPU: {torch.cuda.get_device_name(0)}")

print("[DermLens FastAPI] Loading reference embeddings...")
ref_embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
ref_metadata   = pd.read_csv(METADATA_PATH)
print(f"[DermLens FastAPI] Reference cases: {len(ref_metadata):,}")

# Index images
image_lookup = {}
for folder in IMAGE_DIRS:
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith(".jpg"):
                image_lookup[os.path.splitext(fname)[0]] = os.path.join(folder, fname)
print(f"[DermLens FastAPI] Images indexed: {len(image_lookup):,}")

# Load FAISS
print("[DermLens FastAPI] Loading FAISS index...")
faiss_index = faiss.read_index(INDEX_PATH)

# Load ResNet-50
print("[DermLens FastAPI] Loading ResNet-50 backbone...")
weights = models.ResNet50_Weights.DEFAULT
model   = models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model    = model.to(device).eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Rarity score calibration
class_counts  = ref_metadata["dx"].value_counts()
_raw          = 1.0 / np.sqrt(class_counts)
rarity_scores = (_raw - _raw.min()) / (_raw.max() - _raw.min())

# Age median
ref_metadata["age"] = pd.to_numeric(ref_metadata["age"], errors="coerce")
age_median = float(ref_metadata["age"].median())
ref_metadata["age"] = ref_metadata["age"].fillna(age_median)

print("[DermLens FastAPI] Initialized successfully.\n")

# ═════════════════════════════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════════════════════════════

app = FastAPI(
    title="DermLens CBIR API",
    description="Content-Based Image Retrieval API for Skin Lesion Analysis",
    version="2.0.0",
)

# CORS middleware for React / Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Image Endpoints ──────────────────────────────────────────

@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    path = image_lookup.get(image_id)
    if path and os.path.isfile(path):
        return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")

@app.get("/api/preview/{filename}")
async def get_preview(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if os.path.isfile(path):
        return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Preview file not found")

# ── Inference Endpoint ───────────────────────────────────────

@app.post("/api/analyze")
async def analyze_lesion(
    image: UploadFile = File(...),
    age: Optional[float] = Form(None),
    sex: Optional[str] = Form(None),
    localization: Optional[str] = Form(None),
    top_k: int = Form(5),
):
    t0 = time.time()
    
    if not image.filename:
        raise HTTPException(status_code=400, detail="Empty filename provided")

    # Save preview
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    preview_name = f"{uuid.uuid4().hex}{ext}"
    preview_path = os.path.join(UPLOAD_DIR, preview_name)

    contents = await image.read()
    with open(preview_path, "wb") as f:
        f.write(contents)

    top_k = min(max(top_k, 1), 20)
    q_sex = (sex or "").strip().lower()
    q_loc = (localization or "").strip().lower()

    try:
        # Stage 1: Feature Extraction
        pil_img = Image.open(preview_path).convert("RGB")
        tensor  = img_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        q_emb = emb.cpu().numpy().astype("float32")

        # Stage 2: FAISS Candidate Retrieval (Top-50)
        sims, idxs = faiss_index.search(q_emb, CANDIDATE_K)
        cands = ref_metadata.iloc[idxs[0]].copy()
        cands["visual_similarity"] = sims[0].astype(float)

        # Stage 3: Multimodal Clinical Metadata Fusion
        meta_scores = np.zeros(len(cands))

        if age is not None:
            age_sim = np.maximum(
                0.0,
                1.0 - np.abs(cands["age"].values - age) / 50.0,
            )
            age_sim = np.nan_to_num(age_sim, nan=0.0)
            meta_scores += AGE_WEIGHT * age_sim

        if q_sex:
            sex_sim = (
                cands["sex"].str.strip().str.lower().values == q_sex
            ).astype(float)
            meta_scores += SEX_WEIGHT * sex_sim

        if q_loc:
            loc_sim = (
                cands["localization"].str.strip().str.lower().values == q_loc
            ).astype(float)
            meta_scores += LOCATION_WEIGHT * loc_sim

        cands["metadata_score"] = meta_scores

        # Stage 4: Similarity-Gated Rarity Re-ranking
        cand_rarity = cands["dx"].map(rarity_scores).fillna(0.0).values
        gate = (
            cands["visual_similarity"].values >= SIMILARITY_THRESHOLD
        ).astype(float)
        cands["rarity_adjustment"] = RARITY_BOOST * cand_rarity * gate

        # Stage 5: Final Combined Score
        cands["final_score"] = (
            cands["visual_similarity"]
            + cands["metadata_score"]
            + cands["rarity_adjustment"]
        )

        top = cands.sort_values("final_score", ascending=False).head(top_k)

        results = []
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            results.append({
                "rank": rank,
                "image_id": str(row["image_id"]),
                "lesion_id": str(row.get("lesion_id", "")),
                "diagnosis": str(row["dx"]),
                "diagnosis_full": DX_LABELS.get(row["dx"], row["dx"]),
                "diagnosis_category": DX_CATEGORIES.get(row["dx"], "Benign"),
                "diagnosis_type": str(row.get("dx_type", "histo")),
                "age": None if pd.isna(row["age"]) else round(float(row["age"]), 1),
                "sex": str(row["sex"]) if pd.notna(row.get("sex")) else None,
                "localization": str(row["localization"]) if pd.notna(row.get("localization")) else None,
                "image_url": f"/api/image/{row['image_id']}",
                "scores": {
                    "final_score": round(float(row["final_score"]), 6),
                    "visual_similarity": round(float(row["visual_similarity"]), 6),
                    "metadata_score": round(float(row["metadata_score"]), 6),
                    "rarity_adjustment": round(float(row["rarity_adjustment"]), 6),
                }
            })

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "query": {
                "preview_url": f"/api/preview/{preview_name}",
                "age": age,
                "sex": q_sex or None,
                "localization": q_loc or None,
            },
            "results": results,
            "processing_time_ms": elapsed_ms,
            "pipeline": {
                "model": "ResNet-50 (ImageNet Pretrained)",
                "embedding_dim": int(ref_embeddings.shape[1]),
                "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
                "scoring": "Visual + Metadata + Gated Rarity",
                "candidate_pool": CANDIDATE_K,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "rarity_boost": RARITY_BOOST,
                "device": str(device),
            }
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

# ── Metadata & Statistics ────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    full_df = pd.read_csv(FULL_META_PATH)
    class_dist = full_df["dx"].value_counts().to_dict()
    return {
        "total_images": int(len(full_df)),
        "total_lesions": int(full_df["lesion_id"].nunique()),
        "num_classes": int(full_df["dx"].nunique()),
        "reference_images": int(len(ref_metadata)),
        "faiss_vectors": int(faiss_index.ntotal),
        "embedding_dim": int(ref_embeddings.shape[1]),
        "class_distribution": {
            dx: {"count": int(cnt), "label": DX_LABELS.get(dx, dx), "category": DX_CATEGORIES.get(dx, "Benign")}
            for dx, cnt in class_dist.items()
        },
        "localizations": sorted(full_df["localization"].dropna().unique().tolist()),
        "sex_options": ["male", "female"],
    }

# ── Research Metrics ─────────────────────────────────────────

@app.get("/api/metrics")
async def get_metrics():
    out = {}
    def _read(name):
        p = os.path.join(RESULTS_DIR, name)
        return pd.read_csv(p) if os.path.isfile(p) else None

    df = _read("baseline_metrics_per_class.csv")
    if df is not None:
        out["baseline"] = {
            "per_class": df.to_dict(orient="records"),
            "overall": {c: round(float(df[c].mean()), 6) for c in df.columns if c != "query_dx"}
        }

    df = _read("final_combined_per_class.csv")
    if df is not None:
        out["combined"] = {
            "per_class": df.to_dict(orient="records"),
            "overall": {c: round(float(df[c].mean()), 6) for c in df.columns if c not in ("dx", "n_queries")}
        }

    df = _read("wilcoxon_significance_results.csv")
    if df is not None:
        out["significance"] = df.to_dict(orient="records")

    df = _read("rarity_ablation_results.csv")
    if df is not None:
        out["ablation"] = df.to_dict(orient="records")

    return out

# Serve production React build if dist exists
DIST_DIR = os.path.join(FRONTEND_DIR, "web", "dist")
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static_dist")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
