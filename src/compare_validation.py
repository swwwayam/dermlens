import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# RESULT FILES
# ============================================================

BASELINE_FILE = os.path.join(
    RESULTS_DIR,
    "baseline_metrics_per_query.csv"
)

METADATA_FILE = os.path.join(
    RESULTS_DIR,
    "metadata_validation_results.csv"
)

RARE_FILE = os.path.join(
    RESULTS_DIR,
    "controlled_rare_results.csv"
)

COMBINED_FILE = os.path.join(
    RESULTS_DIR,
    "combined_validation_results.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_comparison.csv"
)


# ============================================================
# HELPER
# ============================================================

def read_query_metrics(path):

    if not os.path.exists(path):

        print(
            f"WARNING: File not found:\n{path}"
        )

        return None

    df = pd.read_csv(path)

    return df


def mean_metric(
    df,
    column
):

    if df is None:
        return np.nan

    if column not in df.columns:
        return np.nan

    return df[column].mean()


# ============================================================
# START
# ============================================================

print(
    "\n========== VALIDATION MODEL COMPARISON ==========\n"
)


# ============================================================
# 1. VISUAL BASELINE
# ============================================================

baseline_df = read_query_metrics(
    BASELINE_FILE
)

baseline = {

    "Model":
        "Visual Baseline",

    "P@5":
        mean_metric(
            baseline_df,
            "precision_at_5"
        ),

    "P@10":
        mean_metric(
            baseline_df,
            "precision_at_10"
        ),

    "NDCG@10":
        mean_metric(
            baseline_df,
            "ndcg_at_10"
        ),

    "MRR":
        mean_metric(
            baseline_df,
            "reciprocal_rank"
        )
}


# ============================================================
# 2. METADATA BEST CONFIGURATION
# ============================================================

metadata_df = read_query_metrics(
    METADATA_FILE
)

metadata_best = None

if metadata_df is not None:

    metadata_best = (
        metadata_df
        .sort_values(
            "ndcg_at_10",
            ascending=False
        )
        .iloc[0]
    )


metadata = {

    "Model":
        "Metadata-Aware",

    "P@5":
        metadata_best[
            "precision_at_5"
        ],

    "P@10":
        metadata_best[
            "precision_at_10"
        ],

    "NDCG@10":
        metadata_best[
            "ndcg_at_10"
        ],

    "MRR":
        metadata_best[
            "mrr"
        ]
}


# ============================================================
# 3. RARE-AWARE
# ============================================================

rare_df = read_query_metrics(
    RARE_FILE
)

rare_best = None

if rare_df is not None:

    # Use the best configuration from the
    # controlled validation-style experiment.
    #
    # Prefer the configuration with the
    # highest NDCG@10.

    rare_best = (
        rare_df
        .sort_values(
            "ndcg_at_10",
            ascending=False
        )
        .iloc[0]
    )


rare = {

    "Model":
        "Rare-Aware",

    "P@5":
        rare_best[
            "precision_at_5"
        ],

    "P@10":
        rare_best[
            "precision_at_10"
        ],

    "NDCG@10":
        rare_best[
            "ndcg_at_10"
        ],

    "MRR":
        rare_best[
            "mrr"
        ]
}


# ============================================================
# 4. COMBINED
# ============================================================

combined_df = read_query_metrics(
    COMBINED_FILE
)

combined_best = None

if combined_df is not None:

    combined_df[
        "balanced_score"
    ] = (

        0.5
        *
        combined_df[
            "ndcg_at_10"
        ]

        +

        0.5
        *
        combined_df[
            "macro_ndcg_at_10"
        ]
    )

    combined_best = (
        combined_df
        .sort_values(
            "balanced_score",
            ascending=False
        )
        .iloc[0]
    )


combined = {

    "Model":
        "Combined Metadata + Rare",

    "P@5":
        combined_best[
            "precision_at_5"
        ],

    "P@10":
        combined_best[
            "precision_at_10"
        ],

    "NDCG@10":
        combined_best[
            "ndcg_at_10"
        ],

    "MRR":
        combined_best[
            "mrr"
        ]
}


# ============================================================
# BUILD TABLE
# ============================================================

comparison = pd.DataFrame([

    baseline,
    metadata,
    rare,
    combined
])


# ============================================================
# CONVERT TO PERCENTAGES
# ============================================================

metric_columns = [

    "P@5",
    "P@10",
    "NDCG@10",
    "MRR"
]

for column in metric_columns:

    comparison[
        column
    ] = (
        comparison[
            column
        ]
        *
        100
    )


# ============================================================
# IMPROVEMENT VS BASELINE
# ============================================================

baseline_ndcg = (
    comparison
    .loc[
        comparison["Model"]
        ==
        "Visual Baseline",
        "NDCG@10"
    ]
    .iloc[0]
)

baseline_p5 = (
    comparison
    .loc[
        comparison["Model"]
        ==
        "Visual Baseline",
        "P@5"
    ]
    .iloc[0]
)

baseline_mrr = (
    comparison
    .loc[
        comparison["Model"]
        ==
        "Visual Baseline",
        "MRR"
    ]
    .iloc[0]
)


comparison[
    "NDCG Improvement"
] = (
    comparison["NDCG@10"]
    -
    baseline_ndcg
)

comparison[
    "P@5 Improvement"
] = (
    comparison["P@5"]
    -
    baseline_p5
)

comparison[
    "MRR Improvement"
] = (
    comparison["MRR"]
    -
    baseline_mrr
)


# ============================================================
# ROUND
# ============================================================

comparison = comparison.round(
    2
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n========== VALIDATION COMPARISON ==========\n"
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nComparison saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# BEST MODELS
# ============================================================

print(
    "\n========== BEST BY OVERALL NDCG ==========\n"
)

best_ndcg = (
    comparison
    .sort_values(
        "NDCG@10",
        ascending=False
    )
    .iloc[0]
)

print(
    f"Model: {best_ndcg['Model']}"
)

print(
    f"NDCG@10: "
    f"{best_ndcg['NDCG@10']:.2f}%"
)


print(
    "\n========== BEST BY P@5 ==========\n"
)

best_p5 = (
    comparison
    .sort_values(
        "P@5",
        ascending=False
    )
    .iloc[0]
)

print(
    f"Model: {best_p5['Model']}"
)

print(
    f"P@5: "
    f"{best_p5['P@5']:.2f}%"
)


print(
    "\n========== COMPLETE ==========\n"
)