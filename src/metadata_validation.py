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

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

FAISS_DIR = os.path.join(
    BASE_DIR,
    "faiss_index"
)

VAL_CSV = os.path.join(
    DATA_DIR,
    "splits",
    "val.csv"
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
    "metadata_validation_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CANDIDATE_K = 50
FINAL_K = 10
BATCH_SIZE = 64

# Metadata weights to test
AGE_WEIGHTS = [
    0.01,
    0.02,
    0.03
]

SEX_WEIGHTS = [
    0.01,
    0.02
]

LOCATION_WEIGHTS = [
    0.01,
    0.02,
    0.03
]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    relevant,
    k
):

    return sum(
        relevant[:k]
    ) / k


def reciprocal_rank(
    relevant
):

    for rank, value in enumerate(
        relevant,
        start=1
    ):

        if value:

            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    relevant,
    k
):

    relevant = relevant[:k]

    dcg = 0.0

    for i, rel in enumerate(
        relevant,
        start=1
    ):

        if rel:

            dcg += (
                1.0
                /
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

        1.0
        /
        np.log2(i + 1)

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
    "\n========== METADATA-AWARE VALIDATION ==========\n"
)

reference_df = pd.read_csv(
    REFERENCE_METADATA
)

val_df = pd.read_csv(
    VAL_CSV
)

index = faiss.read_index(
    INDEX_PATH
)

print(
    f"Reference images: "
    f"{len(reference_df):,}"
)

print(
    f"Validation images: "
    f"{len(val_df):,}"
)

print(
    f"FAISS index size: "
    f"{index.ntotal:,}"
)


# ============================================================
# VERIFY METADATA
# ============================================================

metadata_columns = [
    "age",
    "sex",
    "localization"
]

print(
    "\nChecking metadata..."
)

for column in metadata_columns:

    if column not in reference_df.columns:

        raise ValueError(
            f"Missing column in reference metadata: {column}"
        )

    if column not in val_df.columns:

        raise ValueError(
            f"Missing column in validation data: {column}"
        )

    print(
        f"✓ {column}"
    )


# ============================================================
# CLEAN AGE
# ============================================================

reference_df["age"] = pd.to_numeric(
    reference_df["age"],
    errors="coerce"
)

val_df["age"] = pd.to_numeric(
    val_df["age"],
    errors="coerce"
)

reference_df["age"] = (
    reference_df["age"]
    .fillna(
        reference_df["age"].median()
    )
)

val_df["age"] = (
    val_df["age"]
    .fillna(
        reference_df["age"].median()
    )
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

            image_id = os.path.splitext(
                filename
            )[0]

            image_lookup[
                image_id
            ] = os.path.join(
                folder,
                filename
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
# VALIDATION DATASET
# ============================================================

class ValidationDataset(
    Dataset
):

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

    def __len__(
        self
    ):

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


dataset = ValidationDataset(
    val_df
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
# GENERATE VALIDATION EMBEDDINGS
# ============================================================

print(
    "\nGenerating validation embeddings..."
)

val_embeddings = []

val_image_ids = []

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

        val_embeddings.append(
            embeddings
        )

        val_image_ids.extend(
            image_ids
        )


val_embeddings = np.vstack(
    val_embeddings
)

print(
    f"Validation embedding shape: "
    f"{val_embeddings.shape}"
)


# ============================================================
# FAISS RETRIEVAL
# ============================================================

print(
    "\nRetrieving Top-50 candidates..."
)

similarities, indices = (
    index.search(
        val_embeddings,
        CANDIDATE_K
    )
)

print(
    "✓ Candidate retrieval complete."
)


# ============================================================
# PREPARE CANDIDATES
# ============================================================

candidate_data = []

for query_position in range(
    len(val_image_ids)
):

    query_id = (
        val_image_ids[
            query_position
        ]
    )

    query_row = val_df[
        val_df["image_id"]
        == query_id
    ].iloc[0]

    query_dx = query_row[
        "dx"
    ]

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
    ] = (
        candidate_scores
        .astype(float)
    )

    candidate_data.append({

        "query_id":
            query_id,

        "query_dx":
            query_dx,

        "query_age":
            query_row["age"],

        "query_sex":
            query_row["sex"],

        "query_localization":
            query_row[
                "localization"
            ],

        "candidates":
            candidates
    })


# ============================================================
# METADATA SIMILARITY FUNCTIONS
# ============================================================

def age_similarity(
    query_age,
    candidate_ages
):

    difference = np.abs(
        candidate_ages
        - query_age
    )

    # Maximum useful difference = 50 years.
    # Same age = 1.
    # 50+ years apart = 0.
    score = 1.0 - (
        difference / 50.0
    )

    return np.clip(
        score,
        0.0,
        1.0
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

all_results = []

total_experiments = (
    len(AGE_WEIGHTS)
    *
    len(SEX_WEIGHTS)
    *
    len(LOCATION_WEIGHTS)
)

experiment_number = 0

print(
    "\n========== RUNNING METADATA EXPERIMENTS ==========\n"
)

for age_weight in AGE_WEIGHTS:

    for sex_weight in SEX_WEIGHTS:

        for location_weight in (
            LOCATION_WEIGHTS
        ):

            experiment_number += 1

            print(
                f"Experiment "
                f"{experiment_number}/"
                f"{total_experiments}: "
                f"age={age_weight:.2f}, "
                f"sex={sex_weight:.2f}, "
                f"location={location_weight:.2f}"
            )

            p5_values = []
            p10_values = []
            ndcg_values = []
            mrr_values = []

            per_class = {}


            # ====================================================
            # PROCESS QUERIES
            # ====================================================

            for item in candidate_data:

                query_dx = item[
                    "query_dx"
                ]

                query_age = item[
                    "query_age"
                ]

                query_sex = item[
                    "query_sex"
                ]

                query_location = item[
                    "query_localization"
                ]

                candidates = (
                    item[
                        "candidates"
                    ]
                    .copy()
                )


                # ------------------------------------------------
                # AGE SIMILARITY
                # ------------------------------------------------

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


                # ------------------------------------------------
                # SEX MATCH
                # ------------------------------------------------

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
                )

                sex_score = (
                    sex_score
                    .astype(float)
                )


                # ------------------------------------------------
                # LOCALIZATION MATCH
                # ------------------------------------------------

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
                )

                location_score = (
                    location_score
                    .astype(float)
                )


                # ------------------------------------------------
                # FINAL METADATA-AWARE SCORE
                # ------------------------------------------------

                candidates[
                    "final_score"
                ] = (

                    candidates[
                        "similarity"
                    ].astype(float)

                    +

                    age_weight
                    *
                    age_score

                    +

                    sex_weight
                    *
                    sex_score

                    +

                    location_weight
                    *
                    location_score
                )


                # ------------------------------------------------
                # RANK
                # ------------------------------------------------

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


                retrieved_dx = (
                    top_candidates[
                        "dx"
                    ]
                    .tolist()
                )


                # ------------------------------------------------
                # RELEVANCE
                # ------------------------------------------------

                relevant = [

                    dx == query_dx

                    for dx
                    in retrieved_dx
                ]


                # ------------------------------------------------
                # METRICS
                # ------------------------------------------------

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


                # ------------------------------------------------
                # PER CLASS
                # ------------------------------------------------

                if query_dx not in (
                    per_class
                ):

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


            # ====================================================
            # OVERALL
            # ====================================================

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


            # ====================================================
            # MACRO
            # ====================================================

            macro_p5_values = []
            macro_p10_values = []
            macro_ndcg_values = []
            macro_mrr_values = []


            for cls, values in (
                per_class.items()
            ):

                macro_p5_values.append(
                    np.mean(
                        values["p5"]
                    )
                )

                macro_p10_values.append(
                    np.mean(
                        values["p10"]
                    )
                )

                macro_ndcg_values.append(
                    np.mean(
                        values["ndcg"]
                    )
                )

                macro_mrr_values.append(
                    np.mean(
                        values["mrr"]
                    )
                )


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


            # ====================================================
            # SAVE
            # ====================================================

            all_results.append({

                "age_weight":
                    age_weight,

                "sex_weight":
                    sex_weight,

                "location_weight":
                    location_weight,

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
# DISPLAY
# ============================================================

print(
    "\n========== METADATA VALIDATION RESULTS ==========\n"
)

display_df = (
    results_df
    .copy()
)

metric_columns = [

    "precision_at_5",
    "precision_at_10",
    "ndcg_at_10",
    "mrr",

    "macro_precision_at_5",
    "macro_precision_at_10",
    "macro_ndcg_at_10",
    "macro_mrr"
]

for column in metric_columns:

    display_df[
        column
    ] = (
        display_df[
            column
        ]
        * 100
    ).round(2)


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BASELINE REFERENCE
# ============================================================

print(
    "\n========== REFERENCE BASELINE ==========\n"
)

print(
    "Visual-only baseline:"
)

print(
    "Precision@5:  64.53%"
)

print(
    "Precision@10: 63.85%"
)

print(
    "NDCG@10:      79.30%"
)

print(
    "MRR:          75.64%"
)

print(
    "Macro NDCG@10: 56.54%"
)


# ============================================================
# BEST METADATA CONFIGURATIONS
# ============================================================

results_df[
    "balanced_score"
] = (

    0.5
    *
    results_df[
        "ndcg_at_10"
    ]

    +

    0.5
    *
    results_df[
        "macro_ndcg_at_10"
    ]
)

best = (
    results_df
    .sort_values(
        "balanced_score",
        ascending=False
    )
    .head(5)
)


print(
    "\n========== TOP 5 METADATA CONFIGURATIONS ==========\n"
)

best_display = (
    best
    .copy()
)

for column in metric_columns:

    best_display[
        column
    ] = (
        best_display[
            column
        ]
        * 100
    ).round(2)


print(
    best_display.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

print(
    "\nResults saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\n========== METADATA VALIDATION COMPLETE ==========\n"
)