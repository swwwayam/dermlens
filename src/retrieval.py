import os
import numpy as np
import pandas as pd
import faiss
import torch
from PIL import Image
from torchvision import models, transforms

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

os.makedirs(FAISS_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(
    RESULTS_DIR,
    "train_embeddings.npy"
)

METADATA_PATH = os.path.join(
    RESULTS_DIR,
    "train_embedding_metadata.csv"
)

TEST_CSV = os.path.join(
    DATA_DIR,
    "splits",
    "test.csv"
)

IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2")
]

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "ham10000_resnet.index"
)

# ============================================================
# SETTINGS
# ============================================================

TOP_K = 5

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== FAISS RETRIEVAL SYSTEM ==========\n")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading reference embeddings...")

embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

print(f"Embedding shape: {embeddings.shape}")

# ============================================================
# BUILD FAISS INDEX
# ============================================================
#
# Embeddings were L2-normalized during generation.
# Therefore inner product = cosine similarity.
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index size: {index.ntotal:,}")

faiss.write_index(index, INDEX_PATH)

print(f"Index saved to:")
print(INDEX_PATH)

# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_PATH)

# ============================================================
# IMAGE LOOKUP
# ============================================================

image_lookup = {}

for folder in IMAGE_DIRS:

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            image_id = os.path.splitext(filename)[0]

            image_lookup[image_id] = os.path.join(
                folder,
                filename
            )

# ============================================================
# LOAD RESNET-50
# ============================================================

print("\nLoading ResNet-50...")

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)

model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# QUERY FUNCTION
# ============================================================

def retrieve_similar_cases(image_id, top_k=5):

    if image_id not in image_lookup:
        raise FileNotFoundError(
            f"Image not found: {image_id}"
        )

    image_path = image_lookup[image_id]

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Generate query embedding
    with torch.no_grad():

        embedding = model(image_tensor)

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

    query_embedding = embedding.cpu().numpy().astype("float32")

    # Search FAISS
    similarities, indices = index.search(
        query_embedding,
        top_k + 10
    )

    results = []

    for similarity, idx in zip(
        similarities[0],
        indices[0]
    ):

        result = metadata.iloc[idx].copy()

        result["similarity"] = float(similarity)

        results.append(result)

    results_df = pd.DataFrame(results)

    # Remove query image itself if it exists
    results_df = results_df[
        results_df["image_id"] != image_id
    ]

    return results_df.head(top_k)


# ============================================================
# TEST THE RETRIEVAL SYSTEM
# ============================================================

test_df = pd.read_csv(TEST_CSV)

# Pick the first test image
query_image_id = test_df.iloc[0]["image_id"]

query_diagnosis = test_df.iloc[0]["dx"]

print("\n========== TEST QUERY ==========\n")

print(f"Query image: {query_image_id}")
print(f"Actual diagnosis: {query_diagnosis}")

print("\nRetrieving similar cases...\n")

results = retrieve_similar_cases(
    query_image_id,
    TOP_K
)

print(
    results[
        [
            "image_id",
            "lesion_id",
            "dx",
            "age",
            "sex",
            "localization",
            "similarity"
        ]
    ].to_string(index=False)
)

print("\n========== RETRIEVAL COMPLETE ==========\n")