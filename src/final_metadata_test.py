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

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "final_metadata_test_results.csv"
)


# ============================================================
# LOCKED PARAMETERS
# ============================================================

AGE_WEIGHT = 0.03
SEX_WEIGHT = 0.01
LOCATION_WEIGHT = 0.03

CANDIDATE_K = 50
FINAL_K = 10
BATCH_SIZE = 64


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
# AGE SIMILARITY
# ============================================================

def age_similarity(
    query_age,
    candidate_ages
):

    difference = np.abs(
        candidate_ages - query_age
    )

    score = (
        1.0 -
        difference / 50.0
    )

    return np.clip(
        score,
        0.0,
        1.0
    )


# ============================================================
# START
# ============================================================

print(
    "\n========== FINAL METADATA TEST EVALUATION ==========\n"
)

print(
    "IMPORTANT: PARAMETERS ARE LOCKED."
)

print(
    f"Age weight:          {AGE_WEIGHT}"
)

print(
    f"Sex weight:          {SEX_WEIGHT}"
)

print(
    f"Localization weight: {LOCATION_WEIGHT}"
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
# CHECK COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "dx",
    "age",
    "sex",
    "localization"
]

for column in required_columns:

    if column not in reference_df.columns:
        raise ValueError(
            f"Missing column in reference data: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing column in test data: {column}"
        )

print(
    "\n✓ Required columns verified."
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
    .fillna(reference_age_median)
)

test_df["age"] = (
    test_df["age"]
    .fillna(reference_age_median)
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

        image_path = image_lookup[
            image_id
        ]

        image = Image.open(
            image_path
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

weights = (
    models.ResNet50_Weights.DEFAULT
)

model = models.resnet50(
    weights=weights
)

model.fc = (
    torch.nn.Identity()
)

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
# FAISS RETRIEVAL
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
# EVALUATION
# ============================================================

print(
    "\nApplying LOCKED metadata-aware ranking..."
)

p5_values = []
p10_values = []
ndcg_values = []
mrr_values = []

per_class = {}

query_results = []


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
        test_df["image_id"] == query_id
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

    candidate_indices = indices[
        query_position
    ]

    candidate_scores = similarities[
        query_position
    ]

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


    # ========================================================
    # AGE
    # ========================================================

    candidate_ages = (
        candidates[
            "age"
        ]
        .astype(float)
        .values
    )

    age_score = (
        age_similarity(
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
    # LOCALIZATION
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
    # FINAL SCORE
    # ========================================================

    candidates[
        "final_score"
    ] = (

        candidates[
            "similarity"
        ].astype(float)

        +

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

    p5_values.append(
        p5
    )

    p10_values.append(
        p10
    )

    ndcg_values.append(
        ndcg
    )

    mrr_values.append(
        rr
    )


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


    # ========================================================
    # SAVE QUERY RESULT
    # ========================================================

    query_results.append({

        "query_image_id":
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


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_p5 = np.mean(
    p5_values
)

overall_p10 = np.mean(
    p10_values
)

overall_ndcg = np.mean(
    ndcg_values
)

overall_mrr = np.mean(
    mrr_values
)


# ============================================================
# MACRO RESULTS
# ============================================================

macro_p5 = np.mean([

    np.mean(
        values["p5"]
    )

    for values
    in per_class.values()
])

macro_p10 = np.mean([

    np.mean(
        values["p10"]
    )

    for values
    in per_class.values()
])

macro_ndcg = np.mean([

    np.mean(
        values["ndcg"]
    )

    for values
    in per_class.values()
])

macro_mrr = np.mean([

    np.mean(
        values["mrr"]
    )

    for values
    in per_class.values()
])


# ============================================================
# PER-CLASS TABLE
# ============================================================

per_class_rows = []

for dx, values in per_class.items():

    per_class_rows.append({

        "dx":
            dx,

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


per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_df = (
    per_class_df
    .sort_values(
        "precision_at_5",
        ascending=False
    )
)


# ============================================================
# SAVE QUERY RESULTS
# ============================================================

query_results_df = pd.DataFrame(
    query_results
)

query_output = os.path.join(
    RESULTS_DIR,
    "final_metadata_query_results.csv"
)

query_results_df.to_csv(
    query_output,
    index=False
)


# ============================================================
# SAVE PER-CLASS RESULTS
# ============================================================

class_output = os.path.join(
    RESULTS_DIR,
    "final_metadata_per_class.csv"
)

per_class_df.to_csv(
    class_output,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "age_weight":
        AGE_WEIGHT,

    "sex_weight":
        SEX_WEIGHT,

    "location_weight":
        LOCATION_WEIGHT,

    "precision_at_5":
        overall_p5,

    "precision_at_10":
        overall_p10,

    "ndcg_at_10":
        overall_ndcg,

    "mrr":
        overall_mrr,

    "macro_precision_at_5":
        macro_p5,

    "macro_precision_at_10":
        macro_p10,

    "macro_ndcg_at_10":
        macro_ndcg,

    "macro_mrr":
        macro_mrr
}])


summary.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n========== FINAL METADATA TEST RESULTS ==========\n"
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


print(
    "\n========== PER-CLASS RESULTS ==========\n"
)

display_class = (
    per_class_df.copy()
)

for column in [
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_10",
    "mrr"
]:

    display_class[
        column
    ] = (
        display_class[
            column
        ]
        *
        100
    ).round(2)

print(
    display_class.to_string(
        index=False
    )
)


# ============================================================
# BASELINE COMPARISON
# ============================================================

print(
    "\n========== COMPARISON WITH BASELINE ==========\n"
)

baseline_p5 = 64.53
baseline_p10 = 63.85
baseline_ndcg = 79.30
baseline_mrr = 75.64

print(
    f"P@5 improvement: "
    f"{overall_p5 * 100 - baseline_p5:+.2f} pp"
)

print(
    f"P@10 improvement: "
    f"{overall_p10 * 100 - baseline_p10:+.2f} pp"
)

print(
    f"NDCG@10 improvement: "
    f"{overall_ndcg * 100 - baseline_ndcg:+.2f} pp"
)

print(
    f"MRR improvement: "
    f"{overall_mrr * 100 - baseline_mrr:+.2f} pp"
)


# ============================================================
# FILES
# ============================================================

print(
    "\nResults saved to:"
)

print(
    OUTPUT_PATH
)

print(
    query_output
)

print(
    class_output
)


print(
    "\n========== FINAL METADATA TEST COMPLETE ==========\n"
)