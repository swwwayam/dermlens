"""
DermLens API Server
===================
Thin Flask wrapper around the existing HAM10000 CBIR pipeline.

Integration approach
--------------------
- Loads the pre-built FAISS index from  faiss_index/
- Loads reference embeddings + metadata from  results/
- Uses the same ResNet-50 architecture  (fc → Identity, L2-norm)
- Applies the locked scoring parameters from  src/final_combined_test.py
- Reads research metrics directly from  results/ CSVs

No existing project files are modified.
"""

import os
import sys
import time
import uuid
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
import faiss
import torch
from torchvision import models, transforms

# ── Paths (resolved relative to the project root) ────────────
API_DIR      = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(API_DIR)
BASE_DIR     = os.path.dirname(FRONTEND_DIR)

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

# ── Diagnosis lookup ──────────────────────────────────────────
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

# ── Locked scoring parameters (src/final_combined_test.py) ────
CANDIDATE_K          = 50
SIMILARITY_THRESHOLD = 0.80
RARITY_BOOST         = 0.03
AGE_WEIGHT           = 0.03
SEX_WEIGHT           = 0.01
LOCATION_WEIGHT      = 0.03

# ── Device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[DermLens] Device: {device}")
if torch.cuda.is_available():
    print(f"[DermLens] GPU: {torch.cuda.get_device_name(0)}")

# ── Load reference data ──────────────────────────────────────
print("[DermLens] Loading reference embeddings ...")
ref_embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
ref_metadata   = pd.read_csv(METADATA_PATH)
print(f"[DermLens] Reference set: {len(ref_metadata):,} images, "
      f"dim={ref_embeddings.shape[1]}")

# ── Image lookup ─────────────────────────────────────────────
image_lookup = {}
for folder in IMAGE_DIRS:
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith(".jpg"):
                image_lookup[os.path.splitext(fname)[0]] = \
                    os.path.join(folder, fname)
print(f"[DermLens] Images indexed: {len(image_lookup):,}")

# ── FAISS index ──────────────────────────────────────────────
print("[DermLens] Loading FAISS index ...")
faiss_index = faiss.read_index(INDEX_PATH)
print(f"[DermLens] FAISS vectors: {faiss_index.ntotal:,}")

# ── ResNet-50 (same config as src/retrieval.py) ─────────────
print("[DermLens] Loading ResNet-50 ...")
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

# ── Rarity scores (same formula as src/rare_aware_ranking.py) ─
class_counts  = ref_metadata["dx"].value_counts()
_raw          = 1.0 / np.sqrt(class_counts)
rarity_scores = (_raw - _raw.min()) / (_raw.max() - _raw.min())

# ── Age imputation median ────────────────────────────────────
ref_metadata["age"] = pd.to_numeric(ref_metadata["age"], errors="coerce")
age_median = float(ref_metadata["age"].median())
ref_metadata["age"] = ref_metadata["age"].fillna(age_median)

print("[DermLens] Pipeline ready.\n")

# ═════════════════════════════════════════════════════════════
#  FLASK APPLICATION
# ═════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)
CORS(app)

# ── Page routes ──────────────────────────────────────────────

@app.route("/")
@app.route("/index.html")
def page_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/upload")
@app.route("/upload.html")
def page_upload():
    return send_from_directory(app.static_folder, "upload.html")

@app.route("/dashboard")
@app.route("/dashboard.html")
def page_dashboard():
    return send_from_directory(app.static_folder, "dashboard.html")

@app.route("/about")
@app.route("/about.html")
def page_about():
    return send_from_directory(app.static_folder, "about.html")

@app.route("/css/<path:p>")
def serve_css(p):
    return send_from_directory(
        os.path.join(app.static_folder, "css"), p
    )

@app.route("/js/<path:p>")
def serve_js(p):
    return send_from_directory(
        os.path.join(app.static_folder, "js"), p
    )

# ── Image serving ────────────────────────────────────────────

@app.route("/api/image/<image_id>")
def serve_image(image_id):
    path = image_lookup.get(image_id)
    if path and os.path.isfile(path):
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": f"Image '{image_id}' not found"}), 404

@app.route("/api/preview/<filename>")
def serve_preview(filename):
    # Security: only serve from UPLOAD_DIR
    safe = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if os.path.isfile(path):
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": "Preview not found"}), 404

# ── Core analysis endpoint ───────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    t0 = time.time()

    # ---- Validate upload ----
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    # ---- Save uploaded image ----
    ext = os.path.splitext(f.filename)[1] or ".jpg"
    preview_name = f"{uuid.uuid4().hex}{ext}"
    preview_path = os.path.join(UPLOAD_DIR, preview_name)
    f.save(preview_path)

    # ---- Optional clinical metadata ----
    q_age = request.form.get("age", type=float)
    q_sex = request.form.get("sex", "").strip().lower()
    q_loc = request.form.get("localization", "").strip().lower()
    top_k = min(max(request.form.get("top_k", 5, type=int), 1), 20)

    try:
        # ──── Stage 1: Feature extraction ────────────────────
        img    = Image.open(preview_path).convert("RGB")
        tensor = img_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        q_emb = emb.cpu().numpy().astype("float32")

        # ──── Stage 2: FAISS retrieval (top-50 candidates) ───
        sims, idxs = faiss_index.search(q_emb, CANDIDATE_K)

        cands = ref_metadata.iloc[idxs[0]].copy()
        cands["visual_similarity"] = sims[0].astype(float)

        # ──── Stage 3: Metadata-aware scoring ────────────────
        meta_scores = np.zeros(len(cands))

        if q_age is not None:
            age_sim = np.maximum(
                0.0,
                1.0 - np.abs(cands["age"].values - q_age) / 50.0,
            )
            age_sim = np.nan_to_num(age_sim, nan=0.0)
            meta_scores += AGE_WEIGHT * age_sim
            cands["age_similarity"] = age_sim

        if q_sex:
            sex_sim = (
                cands["sex"].str.strip().str.lower().values == q_sex
            ).astype(float)
            meta_scores += SEX_WEIGHT * sex_sim
            cands["sex_similarity"] = sex_sim

        if q_loc:
            loc_sim = (
                cands["localization"].str.strip().str.lower().values == q_loc
            ).astype(float)
            meta_scores += LOCATION_WEIGHT * loc_sim
            cands["localization_similarity"] = loc_sim

        cands["metadata_score"] = meta_scores

        # ──── Stage 4: Gated rarity boost ────────────────────
        cand_rarity = cands["dx"].map(rarity_scores).fillna(0.0).values
        gate = (
            cands["visual_similarity"].values >= SIMILARITY_THRESHOLD
        ).astype(float)
        cands["rarity_adjustment"] = RARITY_BOOST * cand_rarity * gate

        # ──── Stage 5: Final combined score ──────────────────
        cands["final_score"] = (
            cands["visual_similarity"]
            + cands["metadata_score"]
            + cands["rarity_adjustment"]
        )

        top = cands.sort_values("final_score", ascending=False).head(top_k)

        # ──── Build response ─────────────────────────────────
        results = []
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            results.append({
                "rank":            rank,
                "image_id":        str(row["image_id"]),
                "lesion_id":       str(row.get("lesion_id", "")),
                "diagnosis":       str(row["dx"]),
                "diagnosis_full":  DX_LABELS.get(row["dx"], row["dx"]),
                "diagnosis_category": DX_CATEGORIES.get(row["dx"], ""),
                "diagnosis_type":  str(row.get("dx_type", "")),
                "age":  None if pd.isna(row["age"]) else round(float(row["age"]), 1),
                "sex":  str(row["sex"]) if pd.notna(row.get("sex")) else None,
                "localization": (
                    str(row["localization"])
                    if pd.notna(row.get("localization")) else None
                ),
                "image_url": f"/api/image/{row['image_id']}",
                "scores": {
                    "final_score":       round(float(row["final_score"]), 6),
                    "visual_similarity": round(float(row["visual_similarity"]), 6),
                    "metadata_score":    round(float(row["metadata_score"]), 6),
                    "rarity_adjustment": round(float(row["rarity_adjustment"]), 6),
                },
            })

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return jsonify({
            "query": {
                "preview_url":  f"/api/preview/{preview_name}",
                "age":          q_age,
                "sex":          q_sex or None,
                "localization": q_loc or None,
            },
            "results": results,
            "processing_time_ms": elapsed_ms,
            "pipeline": {
                "model":          "ResNet-50 (ImageNet pretrained)",
                "embedding_dim":  int(ref_embeddings.shape[1]),
                "index_type":     "FAISS IndexFlatIP (cosine similarity)",
                "scoring":        "Visual + Metadata + Gated Rarity",
                "candidates_searched":   CANDIDATE_K,
                "similarity_threshold":  SIMILARITY_THRESHOLD,
                "rarity_boost":          RARITY_BOOST,
                "metadata_weights": {
                    "age":          AGE_WEIGHT,
                    "sex":          SEX_WEIGHT,
                    "localization": LOCATION_WEIGHT,
                },
                "results_returned": len(results),
                "device":           str(device),
            },
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ── Dataset statistics ───────────────────────────────────────

@app.route("/api/stats")
def stats():
    try:
        full_df    = pd.read_csv(FULL_META_PATH)
        class_dist = full_df["dx"].value_counts().to_dict()
        return jsonify({
            "total_images":     int(len(full_df)),
            "total_lesions":    int(full_df["lesion_id"].nunique()),
            "num_classes":      int(full_df["dx"].nunique()),
            "reference_images": int(len(ref_metadata)),
            "faiss_vectors":    int(faiss_index.ntotal),
            "embedding_dim":    int(ref_embeddings.shape[1]),
            "class_distribution": {
                dx: {"count": int(cnt), "label": DX_LABELS.get(dx, dx)}
                for dx, cnt in class_dist.items()
            },
            "localizations": sorted(
                full_df["localization"].dropna().unique().tolist()
            ),
            "sex_options": sorted(
                full_df["sex"].dropna().unique().tolist()
            ),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Research metrics ─────────────────────────────────────────

@app.route("/api/metrics")
def metrics():
    out = {}

    def _read(name):
        p = os.path.join(RESULTS_DIR, name)
        return pd.read_csv(p) if os.path.isfile(p) else None

    # Baseline per-class
    df = _read("baseline_metrics_per_class.csv")
    if df is not None:
        out["baseline"] = {
            "per_class": df.to_dict(orient="records"),
            "overall": {
                c: round(float(df[c].mean()), 6)
                for c in df.columns if c != "query_dx"
            },
        }

    # Final combined per-class
    df = _read("final_combined_per_class.csv")
    if df is not None:
        out["combined"] = {
            "per_class": df.to_dict(orient="records"),
            "overall": {
                c: round(float(df[c].mean()), 6)
                for c in df.columns if c not in ("dx", "n_queries")
            },
        }

    # Wilcoxon significance
    df = _read("wilcoxon_significance_results.csv")
    if df is not None:
        out["significance"] = df.to_dict(orient="records")

    # Rarity ablation
    df = _read("rarity_ablation_results.csv")
    if df is not None:
        out["ablation"] = df.to_dict(orient="records")

    # Bootstrap CIs
    df = _read("bootstrap_confidence_intervals.csv")
    if df is not None:
        out["confidence_intervals"] = df.to_dict(orient="records")

    return jsonify(out)


# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[DermLens] http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
