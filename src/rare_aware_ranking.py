import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

REFERENCE_METADATA = os.path.join(
    RESULTS_DIR,
    "train_embedding_metadata.csv"
)

REFERENCE_EMBEDDINGS = os.path.join(
    RESULTS_DIR,
    "train_embeddings.npy"
)

TEST_CSV = os.path.join(
    DATA_DIR,
    "splits",
    "test.csv"
)

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "ham10000_resnet.index"
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "rare_aware_results.csv"
)

# ============================================================
# SETTINGS
# ============================================================

CANDIDATE_K = 50
FINAL_K = 10

# Controls how strongly rarity affects ranking.
# Start small. We can tune this later.
RARITY_WEIGHT = 0.10


# ============================================================
# LOAD DATA
# ============================================================

print("\n========== RARE-AWARE RETRIEVAL ==========\n")

reference_df = pd.read_csv(
    REFERENCE_METADATA
)

test_df = pd.read_csv(
    TEST_CSV
)

reference_embeddings = np.load(
    REFERENCE_EMBEDDINGS
).astype("float32")


# ============================================================
# LOAD FAISS
# ============================================================

import faiss

index = faiss.read_index(
    INDEX_PATH
)

print(
    f"Reference images: {len(reference_df):,}"
)

print(
    f"Test images: {len(test_df):,}"
)

print(
    f"FAISS index size: {index.ntotal:,}"
)


# ============================================================
# CALCULATE CLASS FREQUENCY
# ============================================================

class_counts = (
    reference_df["dx"]
    .value_counts()
)

print("\n========== REFERENCE CLASS COUNTS ==========\n")

print(class_counts)


# ============================================================
# CALCULATE RARITY SCORES
# ============================================================
#
# Rarity = inverse square-root frequency.
#
# Square-root prevents extremely rare classes from receiving
# an absurdly large boost.
#
# IMPORTANT:
# These values come ONLY from the reference set.
# ============================================================

rarity_scores = (
    1.0 / np.sqrt(class_counts)
)

# Normalize rarity scores to 0-1
rarity_scores = (
    rarity_scores - rarity_scores.min()
) / (
    rarity_scores.max() - rarity_scores.min()
)

print("\n========== RARITY SCORES ==========\n")

for cls in rarity_scores.index:

    print(
        f"{cls:6s} : "
        f"{rarity_scores[cls]:.4f}"
    )


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

    return 0.0


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
# NOTE
# ============================================================
#
# This first experiment uses the EMBEDDINGS ALREADY GENERATED
# for the test queries in baseline_retrieval_results.csv.
#
# However, our saved baseline results only contain the top 10.
# We need Top-50 candidates for re-ranking.
#
# Therefore we regenerate test embeddings here.
# ============================================================

import torch
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# IMAGE LOOKUP
# ============================================================

IMAGE_DIRS = [
    os.path.join(
        DATA_DIR,
        "HAM10000_images_part_1"
    ),
    os.path.join(
        DATA_DIR,
        "HAM10000_images_part_2"
    )
]

image_lookup = {}

for folder in IMAGE_DIRS:

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            image_id = os.path.splitext(
                filename
            )[0]

            image_lookup[image_id] = os.path.join(
                folder,
                filename
            )


# ============================================================
# IMAGE TRANSFORM
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
# TEST DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_id = row["image_id"]

        image = Image.open(
            image_lookup[image_id]
        ).convert("RGB")

        image = transform(image)

        return image, image_id


dataset = TestDataset(
    test_df
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# RESNET-50
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
# RETRIEVAL + RE-RANKING
# ============================================================

all_results = []

print(
    "\nGenerating candidates and applying "
    "rare-aware ranking...\n"
)

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

        # Retrieve 50 candidates
        similarities, indices = index.search(
            embeddings,
            CANDIDATE_K
        )

        for query_id, scores, candidate_indices in zip(
            image_ids,
            similarities,
            indices
        ):

            query_row = test_df[
                test_df["image_id"] == query_id
            ].iloc[0]

            query_dx = query_row["dx"]

            candidates = reference_df.iloc[
                candidate_indices
            ].copy()

            candidates["similarity"] = scores

            # --------------------------------------------
            # RARE-AWARE SCORE
            # --------------------------------------------

            candidates["rarity"] = (
                candidates["dx"]
                .map(rarity_scores)
                .fillna(0)
                .values
            )

            candidates["final_score"] = (
                candidates["similarity"]
                +
                RARITY_WEIGHT *
                candidates["rarity"]
            )

            # --------------------------------------------
            # SORT BY FINAL SCORE
            # --------------------------------------------

            candidates = candidates.sort_values(
                "final_score",
                ascending=False
            )

            # --------------------------------------------
            # TAKE TOP-K
            # --------------------------------------------

            top_candidates = candidates.head(
                FINAL_K
            )

            retrieved_dx = (
                top_candidates["dx"]
                .tolist()
            )

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

            all_results.append({

                "query_image_id":
                    query_id,

                "query_dx":
                    query_dx,

                "retrieved_dx":
                    "|".join(
                        retrieved_dx
                    ),

                "precision_at_5":
                    p5,

                "precision_at_10":
                    p10,

                "ndcg_at_10":
                    n10,

                "mrr":
                    rr
            })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print(
    "\n========== RARE-AWARE RESULTS ==========\n"
)

print(
    f"Precision@5:  "
    f"{results_df['precision_at_5'].mean():.4f}"
)

print(
    f"Precision@10: "
    f"{results_df['precision_at_10'].mean():.4f}"
)

print(
    f"NDCG@10:      "
    f"{results_df['ndcg_at_10'].mean():.4f}"
)

print(
    f"MRR:          "
    f"{results_df['mrr'].mean():.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print(
    "\n========== PER-CLASS RESULTS ==========\n"
)

class_results = (
    results_df
    .groupby("query_dx")
    .agg({
        "precision_at_5": "mean",
        "precision_at_10": "mean",
        "ndcg_at_10": "mean",
        "mrr": "mean"
    })
)

print(
    class_results.to_string()
)


# ============================================================
# MACRO METRICS
# ============================================================

print(
    "\n========== MACRO RESULTS ==========\n"
)

macro_p5 = (
    class_results["precision_at_5"]
    .mean()
)

macro_p10 = (
    class_results["precision_at_10"]
    .mean()
)

macro_ndcg = (
    class_results["ndcg_at_10"]
    .mean()
)

macro_mrr = (
    class_results["mrr"]
    .mean()
)

print(
    f"Macro Precision@5:  {macro_p5:.4f}"
)

print(
    f"Macro Precision@10: {macro_p10:.4f}"
)

print(
    f"Macro NDCG@10:      {macro_ndcg:.4f}"
)

print(
    f"Macro MRR:          {macro_mrr:.4f}"
)


print(
    "\nResults saved to:"
)

print(OUTPUT_PATH)

print(
    "\n========== COMPLETE ==========\n"
)