import os
import numpy as np
import pandas as pd
import torch
import faiss

from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Swayam\Desktop\RS PROJ"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "train.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "test.csv"
)

# HAM10000 images are stored in TWO folders
IMAGE_DIRS = [
    os.path.join(
        BASE_DIR,
        "data",
        "HAM10000_images_part_1"
    ),
    os.path.join(
        BASE_DIR,
        "data",
        "HAM10000_images_part_2"
    )
]

FAISS_INDEX = os.path.join(
    BASE_DIR,
    "faiss_index",
    "ham10000_resnet.index"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "results",
    "pareto_rarity_results.csv"
)

OUTPUT_PLOT = os.path.join(
    BASE_DIR,
    "results",
    "pareto_rarity_curve.png"
)

TOP_K_CANDIDATES = 50
FINAL_K = 10

# Lambda sweep
LAMBDA_VALUES = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10
]

BATCH_SIZE = 64


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("PARETO RARITY TRADE-OFF SWEEP")
print("=" * 60)

print(f"\nDevice: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)

print(
    f"\nReference images: "
    f"{len(train_df):,}"
)

print(
    f"Test images:      "
    f"{len(test_df):,}"
)


# ============================================================
# RARITY SCORES
# ============================================================

class_counts = (
    train_df["dx"]
    .value_counts()
)

# IMPORTANT:
# This is EXACTLY the same rarity definition
# used in rare_aware_ranking.py

rarity_scores = (
    1.0 /
    np.sqrt(class_counts)
)

# Normalize rarity scores to 0-1

rarity_scores = (
    rarity_scores -
    rarity_scores.min()
) / (
    rarity_scores.max() -
    rarity_scores.min()
)

print(
    "\n========== RARITY SCORES ==========\n"
)

for cls in sorted(
    rarity_scores.index,
    key=lambda x: rarity_scores[x]
):

    print(
        f"{cls:6s}: "
        f"{rarity_scores[cls]:.4f}"
    )


# ============================================================
# IMAGE LOOKUP
# ============================================================

print(
    "\nBuilding image lookup..."
)

image_lookup = {}

# Scan both HAM10000 image directories ONCE

for directory in IMAGE_DIRS:

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"Image directory not found:\n{directory}"
        )

    for filename in os.listdir(
        directory
    ):

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        ):

            image_id = os.path.splitext(
                filename
            )[0]

            image_lookup[image_id] = (
                os.path.join(
                    directory,
                    filename
                )
            )


print(
    f"Images indexed: "
    f"{len(image_lookup):,}"
)


# ============================================================
# VERIFY REQUIRED IMAGES
# ============================================================

required_ids = set(
    train_df["image_id"]
).union(
    set(test_df["image_id"])
)

missing_images = (
    required_ids -
    set(image_lookup.keys())
)

if missing_images:

    print(
        f"\nERROR: "
        f"{len(missing_images)} images are missing."
    )

    print(
        "\nExamples:"
    )

    for image_id in list(
        missing_images
    )[:10]:

        print(
            f"  {image_id}"
        )

    raise FileNotFoundError(
        "Some required HAM10000 images "
        "could not be located."
    )

print(
    "✓ All required images located."
)


# ============================================================
# RESNET-50
# ============================================================

print(
    "\nLoading ResNet-50..."
)

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)

# Remove classification head
model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()

# IMPORTANT:
# Must exactly match the preprocessing
# used by the baseline/reference embeddings.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print(
    "ResNet-50 loaded."
)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def generate_embeddings(df):

    embeddings = []

    with torch.no_grad():

        for start in tqdm(
            range(
                0,
                len(df),
                BATCH_SIZE
            ),
            desc="Embedding"
        ):

            batch_df = df.iloc[
                start:
                start + BATCH_SIZE
            ]

            images = []

            for image_id in (
                batch_df["image_id"]
            ):

                path = (
                    image_lookup[
                        image_id
                    ]
                )

                image = Image.open(
                    path
                ).convert(
                    "RGB"
                )

                image = transform(
                    image
                )

                images.append(
                    image
                )

            batch = torch.stack(
                images
            ).to(
                device
            )

            features = model(
                batch
            )

            # L2 normalize embeddings
            # to match FAISS cosine-similarity setup

            features = (
                torch.nn.functional
                .normalize(
                    features,
                    p=2,
                    dim=1
                )
            )

            embeddings.append(
                features.cpu().numpy()
            )

    return np.vstack(
        embeddings
    )


# ============================================================
# TEST EMBEDDINGS
# ============================================================

print(
    "\nGenerating test embeddings..."
)

test_embeddings = (
    generate_embeddings(
        test_df
    )
)

print(
    f"\nTest embedding shape: "
    f"{test_embeddings.shape}"
)


# ============================================================
# LOAD FAISS
# ============================================================

print(
    "\nLoading FAISS index..."
)

index = faiss.read_index(
    FAISS_INDEX
)

print(
    f"FAISS index size: "
    f"{index.ntotal:,}"
)


# ============================================================
# RETRIEVE TOP-50 CANDIDATES
# ============================================================

print(
    "\nRetrieving Top-50 candidates..."
)

similarities, indices = (
    index.search(
        test_embeddings.astype(
            "float32"
        ),
        TOP_K_CANDIDATES
    )
)

print(
    "✓ Candidate retrieval complete."
)


# ============================================================
# METRICS
# ============================================================

def dcg(
    relevances
):

    score = 0.0

    for rank, rel in enumerate(
        relevances,
        start=1
    ):

        score += (
            rel /
            np.log2(
                rank + 1
            )
        )

    return score


def ndcg_at_k(
    relevances,
    k=10
):

    relevances = np.asarray(
        relevances[:k]
    )

    actual_dcg = dcg(
        relevances
    )

    ideal = np.sort(
        relevances
    )[::-1]

    ideal_dcg = dcg(
        ideal
    )

    if ideal_dcg == 0:

        return 0.0

    return (
        actual_dcg /
        ideal_dcg
    )


def precision_at_k(
    relevances,
    k
):

    relevances = np.asarray(
        relevances[:k]
    )

    return np.mean(
        relevances
    )


def reciprocal_rank(
    relevances
):

    for i, rel in enumerate(
        relevances,
        start=1
    ):

        if rel == 1:

            return 1.0 / i

    return 0.0


# ============================================================
# REFERENCE LABELS
# ============================================================

reference_labels = (
    train_df["dx"]
    .to_numpy()
)

test_labels = (
    test_df["dx"]
    .to_numpy()
)

classes = sorted(
    test_df["dx"].unique()
)


# ============================================================
# RUN PARETO SWEEP
# ============================================================

all_results = []

print("\n")
print("=" * 60)
print("RUNNING LAMBDA SWEEP")
print("=" * 60)


for lam in LAMBDA_VALUES:

    print(
        f"\nTesting lambda = "
        f"{lam:.2f}"
    )

    query_p5 = []
    query_p10 = []
    query_ndcg = []
    query_mrr = []

    query_classes = []


    # ========================================================
    # PROCESS EVERY TEST QUERY
    # ========================================================

    for q in range(
        len(test_df)
    ):

        candidate_indices = (
            indices[q]
        )

        candidate_similarities = (
            similarities[q]
        )

        candidate_labels = (
            reference_labels[
                candidate_indices
            ]
        )


        # ====================================================
        # RARITY SCORE OF EACH CANDIDATE
        # ====================================================

        candidate_rarity = np.array(
            [
                rarity_scores[
                    label
                ]

                for label
                in candidate_labels

            ],
            dtype=np.float32
        )


        # ====================================================
        # RARE-AWARE SCORE
        #
        # lambda = 0
        # -> pure visual similarity
        #
        # higher lambda
        # -> increasingly favors rare classes
        # ====================================================

        reranked_scores = (
            candidate_similarities
            +
            lam *
            candidate_rarity
        )


        # ====================================================
        # RERANK
        # ====================================================

        ranking = np.argsort(
            reranked_scores
        )[::-1]

        top_indices = (
            ranking[
                :FINAL_K
            ]
        )

        retrieved_labels = (
            candidate_labels[
                top_indices
            ]
        )


        # ====================================================
        # RELEVANCE
        # ====================================================

        query_label = (
            test_labels[q]
        )

        relevances = (
            retrieved_labels
            ==
            query_label
        ).astype(
            int
        )


        # ====================================================
        # QUERY METRICS
        # ====================================================

        query_p5.append(
            precision_at_k(
                relevances,
                5
            )
        )

        query_p10.append(
            precision_at_k(
                relevances,
                10
            )
        )

        query_ndcg.append(
            ndcg_at_k(
                relevances,
                10
            )
        )

        query_mrr.append(
            reciprocal_rank(
                relevances
            )
        )

        query_classes.append(
            query_label
        )


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    overall_p5 = (
        np.mean(
            query_p5
        ) * 100
    )

    overall_p10 = (
        np.mean(
            query_p10
        ) * 100
    )

    overall_ndcg = (
        np.mean(
            query_ndcg
        ) * 100
    )

    overall_mrr = (
        np.mean(
            query_mrr
        ) * 100
    )


    # ========================================================
    # MACRO METRICS
    # ========================================================

    per_class_ndcg = []

    per_class_p5 = []
    per_class_p10 = []
    per_class_mrr = []


    for cls in classes:

        mask = (
            np.array(
                query_classes
            )
            ==
            cls
        )

        cls_ndcg = np.array(
            query_ndcg
        )[mask]

        cls_p5 = np.array(
            query_p5
        )[mask]

        cls_p10 = np.array(
            query_p10
        )[mask]

        cls_mrr = np.array(
            query_mrr
        )[mask]


        if len(
            cls_ndcg
        ) == 0:

            continue


        per_class_ndcg.append(
            np.mean(
                cls_ndcg
            )
        )

        per_class_p5.append(
            np.mean(
                cls_p5
            )
        )

        per_class_p10.append(
            np.mean(
                cls_p10
            )
        )

        per_class_mrr.append(
            np.mean(
                cls_mrr
            )
        )


    macro_ndcg = (
        np.mean(
            per_class_ndcg
        ) * 100
    )

    macro_p5 = (
        np.mean(
            per_class_p5
        ) * 100
    )

    macro_p10 = (
        np.mean(
            per_class_p10
        ) * 100
    )

    macro_mrr = (
        np.mean(
            per_class_mrr
        ) * 100
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = {

        "lambda":
            lam,

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
    }

    all_results.append(
        result
    )


    print(
        f"P@5 = "
        f"{overall_p5:.2f}% | "
        f"NDCG@10 = "
        f"{overall_ndcg:.2f}% | "
        f"Macro NDCG@10 = "
        f"{macro_ndcg:.2f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("PARETO SWEEP RESULTS")
print("=" * 60)

print(
    results_df[
        [
            "lambda",
            "precision_at_5",
            "precision_at_10",
            "ndcg_at_10",
            "mrr",
            "macro_ndcg_at_10"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PARETO PLOT
# ============================================================

import matplotlib.pyplot as plt

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    results_df[
        "ndcg_at_10"
    ],
    results_df[
        "macro_ndcg_at_10"
    ],
    marker="o"
)


# Annotate every lambda

for _, row in (
    results_df.iterrows()
):

    plt.annotate(
        f"λ={row['lambda']:.2f}",
        (
            row["ndcg_at_10"],
            row["macro_ndcg_at_10"]
        ),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )


plt.xlabel(
    "Overall NDCG@10 (%)"
)

plt.ylabel(
    "Macro NDCG@10 (%)"
)

plt.title(
    "Overall Retrieval Quality vs. "
    "Minority-Class Retrieval"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# BEST POINTS
# ============================================================

best_overall = (
    results_df.loc[
        results_df[
            "ndcg_at_10"
        ].idxmax()
    ]
)

best_macro = (
    results_df.loc[
        results_df[
            "macro_ndcg_at_10"
        ].idxmax()
    ]
)


print("\n")
print("=" * 60)
print("BEST CONFIGURATIONS")
print("=" * 60)


print(
    "\nBest overall NDCG:"
)

print(
    f"Lambda = "
    f"{best_overall['lambda']:.2f}"
)

print(
    f"NDCG@10 = "
    f"{best_overall['ndcg_at_10']:.2f}%"
)


print(
    "\nBest macro NDCG:"
)

print(
    f"Lambda = "
    f"{best_macro['lambda']:.2f}"
)

print(
    f"Macro NDCG@10 = "
    f"{best_macro['macro_ndcg_at_10']:.2f}%"
)


print("\n")
print("=" * 60)
print("PARETO SWEEP COMPLETE")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_CSV}"
)

print(
    f"\nPlot saved to:\n"
    f"{OUTPUT_PLOT}"
)