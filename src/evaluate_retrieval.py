import os
import numpy as np
import pandas as pd
import torch
import faiss

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

TEST_CSV = os.path.join(
    DATA_DIR,
    "splits",
    "test.csv"
)

TRAIN_METADATA = os.path.join(
    RESULTS_DIR,
    "train_embedding_metadata.csv"
)

TRAIN_EMBEDDINGS = os.path.join(
    RESULTS_DIR,
    "train_embeddings.npy"
)

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "ham10000_resnet.index"
)

IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2")
]

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "baseline_retrieval_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 10
BATCH_SIZE = 64


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== BASELINE RETRIEVAL EVALUATION ==========\n")

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# IMAGE LOOKUP
# ============================================================

print("\nBuilding image lookup...")

image_lookup = {}

for folder in IMAGE_DIRS:

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            image_id = os.path.splitext(filename)[0]

            image_lookup[image_id] = os.path.join(
                folder,
                filename
            )

print(f"Images indexed: {len(image_lookup):,}")


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_CSV)

print(f"Test images: {len(test_df):,}")


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

reference_df = pd.read_csv(TRAIN_METADATA)

reference_embeddings = np.load(
    TRAIN_EMBEDDINGS
).astype("float32")

print(f"Reference images: {len(reference_df):,}")
print(f"Embedding dimension: {reference_embeddings.shape[1]}")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print(f"FAISS index size: {index.ntotal:,}")


# ============================================================
# IMAGE DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class TestDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_id = row["image_id"]

        image_path = image_lookup[image_id]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        return image, image_id


dataset = TestDataset(test_df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# LOAD RESNET-50
# ============================================================

print("\nLoading ResNet-50...")

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)

model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()


# ============================================================
# GENERATE TEST EMBEDDINGS + RETRIEVE
# ============================================================

all_results = []

print("\nGenerating test embeddings and retrieving cases...\n")

with torch.no_grad():

    for images, image_ids in tqdm(
        loader,
        desc="Evaluating"
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        embeddings = model(images)

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )

        embeddings = embeddings.cpu().numpy().astype(
            "float32"
        )

        similarities, indices = index.search(
            embeddings,
            TOP_K
        )

        for query_id, scores, retrieved_indices in zip(
            image_ids,
            similarities,
            indices
        ):

            query_row = test_df[
                test_df["image_id"] == query_id
            ].iloc[0]

            query_dx = query_row["dx"]

            retrieved_rows = reference_df.iloc[
                retrieved_indices
            ]

            retrieved_dx = retrieved_rows[
                "dx"
            ].tolist()

            all_results.append({
                "query_image_id": query_id,
                "query_dx": query_dx,
                "retrieved_image_ids": "|".join(
                    retrieved_rows["image_id"].tolist()
                ),
                "retrieved_dx": "|".join(
                    retrieved_dx
                ),
                "similarities": "|".join(
                    [f"{x:.6f}" for x in scores]
                )
            })


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nRaw retrieval results saved to:")

print(OUTPUT_PATH)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def precision_at_k(relevant, k):

    relevant = relevant[:k]

    return sum(relevant) / k


def reciprocal_rank(relevant):

    for rank, value in enumerate(
        relevant,
        start=1
    ):

        if value:

            return 1 / rank

    return 0


def ndcg_at_k(relevant, k):

    relevant = relevant[:k]

    dcg = 0.0

    for i, rel in enumerate(
        relevant,
        start=1
    ):

        if rel:

            dcg += 1 / np.log2(i + 1)

    num_relevant = sum(relevant)

    ideal = min(
        num_relevant,
        k
    )

    if ideal == 0:

        return 0.0

    idcg = sum(
        1 / np.log2(i + 1)
        for i in range(1, ideal + 1)
    )

    return dcg / idcg


# ============================================================
# CALCULATE METRICS
# ============================================================

precision5 = []
precision10 = []
ndcg10 = []
mrr = []

per_query_results = []

for _, row in results_df.iterrows():

    query_dx = row["query_dx"]

    retrieved_dx = row[
        "retrieved_dx"
    ].split("|")

    relevant = [
        dx == query_dx
        for dx in retrieved_dx
    ]

    p5 = precision_at_k(
        relevant,
        5
    )

    p10 = precision_at_k(
        relevant,
        10
    )

    n10 = ndcg_at_k(
        relevant,
        10
    )

    rr = reciprocal_rank(
        relevant
    )

    precision5.append(p5)
    precision10.append(p10)
    ndcg10.append(n10)
    mrr.append(rr)

    per_query_results.append({
        "query_image_id": row[
            "query_image_id"
        ],
        "query_dx": query_dx,
        "precision_at_5": p5,
        "precision_at_10": p10,
        "ndcg_at_10": n10,
        "reciprocal_rank": rr
    })


# ============================================================
# OVERALL RESULTS
# ============================================================

evaluation_df = pd.DataFrame(
    per_query_results
)

evaluation_output = os.path.join(
    RESULTS_DIR,
    "baseline_metrics_per_query.csv"
)

evaluation_df.to_csv(
    evaluation_output,
    index=False
)

print("\n========== BASELINE RESULTS ==========\n")

print(
    f"Precision@5:  {np.mean(precision5):.4f}"
)

print(
    f"Precision@10: {np.mean(precision10):.4f}"
)

print(
    f"NDCG@10:      {np.mean(ndcg10):.4f}"
)

print(
    f"MRR:          {np.mean(mrr):.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n========== PER-CLASS RESULTS ==========\n")

class_results = (
    evaluation_df
    .groupby("query_dx")
    .agg({
        "precision_at_5": "mean",
        "precision_at_10": "mean",
        "ndcg_at_10": "mean",
        "reciprocal_rank": "mean"
    })
    .sort_values(
        "ndcg_at_10",
        ascending=False
    )
)

print(class_results.to_string())


# ============================================================
# SAVE PER-CLASS RESULTS
# ============================================================

class_output = os.path.join(
    RESULTS_DIR,
    "baseline_metrics_per_class.csv"
)

class_results.to_csv(
    class_output
)

print("\nPer-query metrics saved to:")

print(evaluation_output)

print("\nPer-class metrics saved to:")

print(class_output)

print("\n========== EVALUATION COMPLETE ==========\n")