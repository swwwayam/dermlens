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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

QUERY_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "final_combined_query_results.csv"
)

PER_CLASS_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "final_combined_per_class.csv"
)


# ============================================================
# LOCKED FINAL PARAMETERS
# ============================================================

CANDIDATE_K = 50
FINAL_K = 10
BATCH_SIZE = 64

# Locked from validation
SIMILARITY_THRESHOLD = 0.80
RARITY_BOOST = 0.03

AGE_WEIGHT = 0.03
SEX_WEIGHT = 0.01
LOCATION_WEIGHT = 0.03


# ============================================================
# METRICS
# ============================================================

def precision_at_k(relevant, k):

    return sum(
        relevant[:k]
    ) / k


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

            dcg += (
                1.0 /
                np.log2(i + 1)
            )

    number_relevant = sum(
        relevant
    )

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
# HEADER
# ============================================================

print(
    "\n========== FINAL COMBINED METADATA + RARE TEST ==========\n"
)

print(
    "IMPORTANT: PARAMETERS ARE LOCKED.\n"
)

print(
    f"Similarity threshold: {SIMILARITY_THRESHOLD}"
)

print(
    f"Rarity boost:         {RARITY_BOOST}"
)

print(
    f"Age weight:           {AGE_WEIGHT}"
)

print(
    f"Sex weight:           {SEX_WEIGHT}"
)

print(
    f"Localization weight:  {LOCATION_WEIGHT}"
)


# ============================================================
# LOAD DATA
# ============================================================

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
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "age",
    "sex",
    "localization",
    "dx"
]

for column in required_columns:

    if column not in reference_df.columns:

        raise ValueError(
            f"Missing column in reference metadata: {column}"
        )

    if column not in test_df.columns:

        raise ValueError(
            f"Missing column in test data: {column}"
        )

print(
    "\n✓ Required metadata columns found."
)


# ============================================================
# CLEAN AGE
# ============================================================

reference_df["age"] = pd.to_numeric(
    reference_df["age"],
    errors="coerce"
)

test_df["age"] = pd.to_numeric(
    test_df["age"],
    errors="coerce"
)

reference_age_median = (
    reference_df["age"].median()
)

reference_df["age"] = (
    reference_df["age"]
    .fillna(
        reference_age_median
    )
)

test_df["age"] = (
    test_df["age"]
    .fillna(
        reference_age_median
    )
)


# ============================================================
# RARITY SCORES
# ============================================================

class_counts = (
    reference_df["dx"]
    .value_counts()
)

rarity_scores = (
    1.0 /
    np.sqrt(class_counts)
)

rarity_scores = (
    rarity_scores
    - rarity_scores.min()
) / (
    rarity_scores.max()
    - rarity_scores.min()
)

print(
    "\n========== RARITY SCORES ==========\n"
)

for cls in rarity_scores.index:

    print(
        f"{cls:6s}: "
        f"{rarity_scores[cls]:.4f}"
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

    for filename in os.listdir(
        folder
    ):

        if filename.lower().endswith(
            ".jpg"
        ):

            image_id = (
                os.path.splitext(
                    filename
                )[0]
            )

            image_lookup[
                image_id
            ] = os.path.join(
                folder,
                filename
            )

print(
    f"Images indexed: "
    f"{len(image_lookup):,}"
)


# ============================================================
# VERIFY TEST IMAGES
# ============================================================

missing_images = []

for image_id in test_df[
    "image_id"
].values:

    if image_id not in image_lookup:

        missing_images.append(
            image_id
        )

if missing_images:

    raise FileNotFoundError(
        f"Missing {len(missing_images)} "
        f"test images. First few: "
        f"{missing_images[:10]}"
    )

print(
    "✓ All test images located."
)


# ============================================================
# TRANSFORM
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

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

    def __len__(self):

        return len(
            self.df
        )

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
            image_lookup[
                image_id
            ]
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

print(
    "ResNet-50 loaded."
)


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
            torch.nn.functional
            .normalize(
                embeddings,
                p=2,
                dim=1
            )
        )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype(
                "float32"
            )
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
# FAISS TOP-50 RETRIEVAL
# ============================================================

print(
    "\nRetrieving Top-50 candidates..."
)

similarities, indices = (
    index.search(
        test_embeddings,
        CANDIDATE_K
    )
)

print(
    "✓ Candidate retrieval complete."
)


# ============================================================
# AGE SIMILARITY
# ============================================================

def calculate_age_similarity(
    query_age,
    candidate_ages
):

    difference = np.abs(
        candidate_ages
        - query_age
    )

    score = (
        1.0
        -
        difference / 50.0
    )

    return np.clip(
        score,
        0.0,
        1.0
    )


# ============================================================
# FINAL RANKING
# ============================================================

print(
    "\nApplying LOCKED combined ranking..."
)

query_results = []

per_class = {}


for query_position in tqdm(
    range(
        len(test_image_ids)
    ),
    desc="Ranking"
):

    query_id = test_image_ids[
        query_position
    ]

    query_row = test_df[
        test_df["image_id"]
        == query_id
    ].iloc[0]

    query_dx = query_row[
        "dx"
    ]

    query_age = query_row[
        "age"
    ]

    query_sex = query_row[
        "sex"
    ]

    query_location = query_row[
        "localization"
    ]


    # ========================================================
    # CANDIDATES
    # ========================================================

    candidate_indices = (
        indices[
            query_position
        ]
    )

    candidate_scores = (
        similarities[
            query_position
        ]
    )

    candidates = (
        reference_df
        .iloc[
            candidate_indices
        ]
        .copy()
    )

    candidates[
        "similarity"
    ] = candidate_scores.astype(
        float
    )

    candidates[
        "rarity"
    ] = (
        candidates["dx"]
        .map(
            rarity_scores
        )
        .fillna(0)
        .astype(float)
    )


    # ========================================================
    # AGE
    # ========================================================

    candidate_ages = (
        candidates["age"]
        .astype(float)
        .values
    )

    age_score = (
        calculate_age_similarity(
            query_age,
            candidate_ages
        )
    )


    # ========================================================
    # SEX
    # ========================================================

    sex_score = (

        candidates[
            "sex"
        ]
        .astype(str)
        .values

        ==

        str(
            query_sex
        )

    ).astype(float)


    # ========================================================
    # LOCATION
    # ========================================================

    location_score = (

        candidates[
            "localization"
        ]
        .astype(str)
        .values

        ==

        str(
            query_location
        )

    ).astype(float)


    # ========================================================
    # METADATA SCORE
    # ========================================================

    metadata_score = (

        AGE_WEIGHT
        *
        age_score

        +

        SEX_WEIGHT
        *
        sex_score

        +

        LOCATION_WEIGHT
        *
        location_score
    )


    # ========================================================
    # CONTROLLED RARITY
    # ========================================================

    eligible = (

        candidates[
            "similarity"
        ].astype(float)

        >=

        SIMILARITY_THRESHOLD
    )

    rarity_adjustment = (

        RARITY_BOOST

        *

        candidates[
            "rarity"
        ].astype(float)
    )


    # ========================================================
    # FINAL COMBINED SCORE
    # ========================================================

    candidates[
        "final_score"
    ] = (

        candidates[
            "similarity"
        ].astype(float)

        +

        metadata_score

        +

        np.where(
            eligible,
            rarity_adjustment,
            0.0
        )
    )


    # ========================================================
    # RANK
    # ========================================================

    candidates = (
        candidates
        .sort_values(
            "final_score",
            ascending=False
        )
    )

    top_candidates = (
        candidates
        .head(
            FINAL_K
        )
    )


    # ========================================================
    # RELEVANCE
    # ========================================================

    retrieved_dx = (
        top_candidates[
            "dx"
        ].tolist()
    )

    relevant = [

        dx == query_dx

        for dx
        in retrieved_dx
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

    query_results.append({

        "image_id":
            query_id,

        "query_dx":
            query_dx,

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
# QUERY RESULTS DATAFRAME
# ============================================================

query_results_df = pd.DataFrame(
    query_results
)

query_results_df.to_csv(
    QUERY_OUTPUT_PATH,
    index=False
)


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_p5 = (
    query_results_df[
        "precision_at_5"
    ].mean()
)

overall_p10 = (
    query_results_df[
        "precision_at_10"
    ].mean()
)

overall_ndcg = (
    query_results_df[
        "ndcg_at_10"
    ].mean()
)

overall_mrr = (
    query_results_df[
        "mrr"
    ].mean()
)


# ============================================================
# MACRO RESULTS
# ============================================================

macro_p5_values = []
macro_p10_values = []
macro_ndcg_values = []
macro_mrr_values = []

per_class_rows = []

for cls, values in (
    per_class.items()
):

    cls_p5 = np.mean(
        values["p5"]
    )

    cls_p10 = np.mean(
        values["p10"]
    )

    cls_ndcg = np.mean(
        values["ndcg"]
    )

    cls_mrr = np.mean(
        values["mrr"]
    )

    n_queries = len(
        values["p5"]
    )

    macro_p5_values.append(
        cls_p5
    )

    macro_p10_values.append(
        cls_p10
    )

    macro_ndcg_values.append(
        cls_ndcg
    )

    macro_mrr_values.append(
        cls_mrr
    )

    per_class_rows.append({

        "dx":
            cls,

        "n_queries":
            n_queries,

        "precision_at_5":
            cls_p5,

        "precision_at_10":
            cls_p10,

        "ndcg_at_10":
            cls_ndcg,

        "mrr":
            cls_mrr
    })


macro_p5 = np.mean(
    macro_p5_values
)

macro_p10 = np.mean(
    macro_p10_values
)

macro_ndcg = np.mean(
    macro_ndcg_values
)

macro_mrr = np.mean(
    macro_mrr_values
)


# ============================================================
# PER-CLASS OUTPUT
# ============================================================

per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_df.to_csv(
    PER_CLASS_OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n========== FINAL COMBINED TEST RESULTS ==========\n"
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

display_per_class = (
    per_class_df.copy()
)

for column in [
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_10",
    "mrr"
]:

    display_per_class[
        column
    ] = (
        display_per_class[
            column
        ]
        *
        100
    ).round(2)

print(
    display_per_class.to_string(
        index=False
    )
)


# ============================================================
# OUTPUT FILES
# ============================================================

print(
    "\nQuery-level results saved to:"
)

print(
    QUERY_OUTPUT_PATH
)

print(
    "\nPer-class results saved to:"
)

print(
    PER_CLASS_OUTPUT_PATH
)


print(
    "\n========== FINAL COMBINED TEST COMPLETE ==========\n"
)