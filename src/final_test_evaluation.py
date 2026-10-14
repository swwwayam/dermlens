import os
import numpy as np
import pandas as pd
import faiss
import torch

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

REFERENCE_METADATA = os.path.join(
    RESULTS_DIR,
    "train_embedding_metadata.csv"
)

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "ham10000_resnet.index"
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "final_test_results.csv"
)


# ============================================================
# LOCKED PARAMETERS
# ============================================================
#
# These were selected using VALIDATION ONLY.
#
# DO NOT change them after seeing test results.
# ============================================================

SIMILARITY_THRESHOLD = 0.80
RARITY_BOOST = 0.02

CANDIDATE_K = 50
FINAL_K = 10
BATCH_SIZE = 64


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def precision_at_k(relevant, k):

    return sum(relevant[:k]) / k


def reciprocal_rank(relevant):

    for rank, value in enumerate(
        relevant,
        start=1
    ):

        if value:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(relevant, k):

    relevant = relevant[:k]

    dcg = 0.0

    for i, rel in enumerate(
        relevant,
        start=1
    ):

        if rel:
            dcg += 1.0 / np.log2(i + 1)

    number_relevant = sum(relevant)

    ideal_relevant = min(
        number_relevant,
        k
    )

    if ideal_relevant == 0:

        return 0.0

    idcg = sum(
        1.0 / np.log2(i + 1)
        for i in range(
            1,
            ideal_relevant + 1
        )
    )

    return dcg / idcg


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n========== FINAL TEST EVALUATION ==========\n"
)

print(
    "IMPORTANT: Parameters are LOCKED."
)

print(
    f"Similarity threshold: "
    f"{SIMILARITY_THRESHOLD}"
)

print(
    f"Rarity boost: "
    f"{RARITY_BOOST}"
)

reference_df = pd.read_csv(
    REFERENCE_METADATA
)

test_df = pd.read_csv(
    TEST_CSV
)

index = faiss.read_index(
    INDEX_PATH
)

print(
    f"\nReference images: "
    f"{len(reference_df):,}"
)

print(
    f"Test images: "
    f"{len(test_df):,}"
)

print(
    f"FAISS index size: "
    f"{index.ntotal:,}"
)


# ============================================================
# RARITY SCORES
# ============================================================

class_counts = (
    reference_df["dx"]
    .value_counts()
)

rarity_scores = (
    1.0 / np.sqrt(class_counts)
)

rarity_scores = (
    rarity_scores - rarity_scores.min()
) / (
    rarity_scores.max() - rarity_scores.min()
)

print(
    "\nReference class counts:"
)

print(
    class_counts
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nDevice: {device}"
)

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

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# TEST DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        image_id = row[
            "image_id"
        ]

        image = Image.open(
            image_lookup[image_id]
        ).convert(
            "RGB"
        )

        image = transform(
            image
        )

        return (
            image,
            image_id
        )


dataset = TestDataset(
    test_df
)

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

print(
    "\nLoading ResNet-50..."
)

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)

model.fc = torch.nn.Identity()

model = model.to(
    device
)

model.eval()


# ============================================================
# GENERATE TEST EMBEDDINGS
# ============================================================

print(
    "\nGenerating test embeddings..."
)

test_embeddings = []

test_image_ids = []

with torch.no_grad():

    for images, image_ids in tqdm(

        loader,

        desc="Embedding"

    ):

        images = images.to(
            device,
            non_blocking=True
        )

        embeddings = model(
            images
        )

        embeddings = (
            torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )
        )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype("float32")
        )

        test_embeddings.append(
            embeddings
        )

        test_image_ids.extend(
            image_ids
        )


test_embeddings = np.vstack(
    test_embeddings
)

print(
    f"Test embedding shape: "
    f"{test_embeddings.shape}"
)


# ============================================================
# RETRIEVE TOP-50
# ============================================================

print(
    "\nRetrieving Top-50 candidates..."
)

similarities, indices = index.search(
    test_embeddings,
    CANDIDATE_K
)

print(
    "✓ Candidate retrieval complete."
)


# ============================================================
# EVALUATE
# ============================================================

all_results = []

per_class = {}

print(
    "\nApplying LOCKED rare-aware ranking..."
)

for query_position in tqdm(
    range(len(test_image_ids)),
    desc="Ranking"
):

    query_id = test_image_ids[
        query_position
    ]

    query_row = test_df[
        test_df["image_id"] == query_id
    ].iloc[0]

    query_dx = query_row[
        "dx"
    ]

    candidate_indices = indices[
        query_position
    ]

    candidate_scores = similarities[
        query_position
    ]

    candidates = reference_df.iloc[
        candidate_indices
    ].copy()

    candidates["similarity"] = (
        candidate_scores.astype(float)
    )

    candidates["rarity"] = (
        candidates["dx"]
        .map(rarity_scores)
        .fillna(0)
        .astype(float)
    )


    # ========================================================
    # CONTROLLED RARE-AWARE SCORE
    # ========================================================

    eligible = (
        candidates["similarity"].astype(float)
        >= SIMILARITY_THRESHOLD
    )

    rarity_adjustment = (
        RARITY_BOOST
        *
        candidates["rarity"].astype(float)
    )

    candidates["final_score"] = (

        candidates["similarity"].astype(float)

        +

        np.where(
            eligible,
            rarity_adjustment,
            0.0
        )
    )


    # ========================================================
    # FINAL TOP-10
    # ========================================================

    candidates = candidates.sort_values(
        "final_score",
        ascending=False
    )

    top_candidates = candidates.head(
        FINAL_K
    )


    retrieved_dx = (
        top_candidates["dx"].tolist()
    )

    retrieved_ids = (
        top_candidates["image_id"].tolist()
    )

    retrieved_scores = (
        top_candidates["final_score"]
        .tolist()
    )


    # ========================================================
    # RELEVANCE
    # ========================================================

    relevant = [

        dx == query_dx

        for dx in retrieved_dx
    ]


    # ========================================================
    # METRICS
    # ========================================================

    p5 = precision_at_k(
        relevant,
        5
    )

    p10 = precision_at_k(
        relevant,
        10
    )

    ndcg = ndcg_at_k(
        relevant,
        10
    )

    rr = reciprocal_rank(
        relevant
    )


    # ========================================================
    # SAVE QUERY RESULT
    # ========================================================

    all_results.append({

        "query_image_id":
            query_id,

        "query_dx":
            query_dx,

        "retrieved_image_ids":
            "|".join(
                retrieved_ids
            ),

        "retrieved_dx":
            "|".join(
                retrieved_dx
            ),

        "retrieved_scores":
            "|".join(
                f"{x:.6f}"
                for x in retrieved_scores
            ),

        "precision_at_5":
            p5,

        "precision_at_10":
            p10,

        "ndcg_at_10":
            ndcg,

        "mrr":
            rr
    })


    # ========================================================
    # PER CLASS
    # ========================================================

    if query_dx not in per_class:

        per_class[
            query_dx
        ] = {

            "p5": [],
            "p10": [],
            "ndcg": [],
            "mrr": []
        }


    per_class[
        query_dx
    ]["p5"].append(
        p5
    )

    per_class[
        query_dx
    ]["p10"].append(
        p10
    )

    per_class[
        query_dx
    ]["ndcg"].append(
        ndcg
    )

    per_class[
        query_dx
    ]["mrr"].append(
        rr
    )


# ============================================================
# RESULTS DATAFRAME
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

overall_p5 = results_df[
    "precision_at_5"
].mean()

overall_p10 = results_df[
    "precision_at_10"
].mean()

overall_ndcg = results_df[
    "ndcg_at_10"
].mean()

overall_mrr = results_df[
    "mrr"
].mean()


# ============================================================
# MACRO RESULTS
# ============================================================

class_results = []

for cls, values in per_class.items():

    class_results.append({

        "dx": cls,

        "precision_at_5":
            np.mean(
                values["p5"]
            ),

        "precision_at_10":
            np.mean(
                values["p10"]
            ),

        "ndcg_at_10":
            np.mean(
                values["ndcg"]
            ),

        "mrr":
            np.mean(
                values["mrr"]
            )
    })


class_results_df = pd.DataFrame(
    class_results
)

class_results_df = (
    class_results_df
    .sort_values(
        "ndcg_at_10",
        ascending=False
    )
)

macro_p5 = (
    class_results_df[
        "precision_at_5"
    ].mean()
)

macro_p10 = (
    class_results_df[
        "precision_at_10"
    ].mean()
)

macro_ndcg = (
    class_results_df[
        "ndcg_at_10"
    ].mean()
)

macro_mrr = (
    class_results_df[
        "mrr"
    ].mean()
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n========== FINAL TEST RESULTS ==========\n"
)

print(
    f"Precision@5:  "
    f"{overall_p5 * 100:.2f}%"
)

print(
    f"Precision@10: "
    f"{overall_p10 * 100:.2f}%"
)

print(
    f"NDCG@10:      "
    f"{overall_ndcg * 100:.2f}%"
)

print(
    f"MRR:          "
    f"{overall_mrr * 100:.2f}%"
)


print(
    "\n========== MACRO RESULTS ==========\n"
)

print(
    f"Macro Precision@5:  "
    f"{macro_p5 * 100:.2f}%"
)

print(
    f"Macro Precision@10: "
    f"{macro_p10 * 100:.2f}%"
)

print(
    f"Macro NDCG@10:      "
    f"{macro_ndcg * 100:.2f}%"
)

print(
    f"Macro MRR:          "
    f"{macro_mrr * 100:.2f}%"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print(
    "\n========== PER-CLASS RESULTS ==========\n"
)

display_class = class_results_df.copy()

for column in [
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_10",
    "mrr"
]:

    display_class[column] = (
        display_class[column]
        * 100
    ).round(2)

print(
    display_class.to_string(
        index=False
    )
)


print(
    "\nQuery-level results saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\n========== FINAL TEST COMPLETE ==========\n"
)