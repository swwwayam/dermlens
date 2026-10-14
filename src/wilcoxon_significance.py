import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Swayam\Desktop\RS PROJ"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "wilcoxon_significance_results.csv"
)


# ============================================================
# INPUT FILES
# ============================================================

MODEL_FILES = {

    "Visual Baseline":
        "baseline_metrics_per_query.csv",

    "Rare-Aware":
        "final_test_results.csv",

    "Metadata-Aware":
        "final_metadata_query_results.csv",

    "Combined Metadata + Rare":
        "final_combined_query_results.csv"
}


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_METRIC = "ndcg_at_10"

ALPHA = 0.05


# ============================================================
# HELPERS
# ============================================================

def find_class_column(df):

    possible_columns = [
        "query_dx",
        "dx",
        "class",
        "label"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


def find_query_id_column(df):

    possible_columns = [
        "query_image_id",
        "image_id",
        "query_id"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


# ============================================================
# START
# ============================================================

print("=" * 75)
print(
    "WILCOXON SIGNED-RANK SIGNIFICANCE TEST"
)
print("=" * 75)

print(
    "\nTarget metric: NDCG@10"
)

print(
    "Test: Paired Wilcoxon signed-rank"
)

print(
    f"Significance level: α = {ALPHA}"
)


# ============================================================
# LOAD ALL MODELS
# ============================================================

data = {}


for model_name, filename in (
    MODEL_FILES.items()
):

    filepath = os.path.join(
        RESULTS_DIR,
        filename
    )


    print("\n")
    print("-" * 75)
    print(
        f"Loading: {model_name}"
    )
    print("-" * 75)


    if not os.path.exists(
        filepath
    ):

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{filepath}"
        )


    df = pd.read_csv(
        filepath
    )


    print(
        f"Queries loaded: "
        f"{len(df):,}"
    )


    # --------------------------------------------------------
    # Check NDCG column
    # --------------------------------------------------------

    if TARGET_METRIC not in df.columns:

        raise ValueError(
            f"\n{TARGET_METRIC} not found "
            f"in {model_name}.\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


    # --------------------------------------------------------
    # Query ID
    # --------------------------------------------------------

    query_id_column = (
        find_query_id_column(df)
    )


    if query_id_column is None:

        raise ValueError(
            f"\nCould not identify query "
            f"ID column in {model_name}."
        )


    print(
        f"Query ID column: "
        f"{query_id_column}"
    )


    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    model_data = df[
        [
            query_id_column,
            TARGET_METRIC
        ]
    ].copy()


    model_data = model_data.rename(
        columns={
            query_id_column:
                "query_id"
        }
    )


    # --------------------------------------------------------
    # Remove duplicate query IDs
    # --------------------------------------------------------

    if model_data[
        "query_id"
    ].duplicated().any():

        print(
            "⚠ Duplicate query IDs detected."
        )

        model_data = (
            model_data
            .drop_duplicates(
                subset="query_id"
            )
        )


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    model_data[
        TARGET_METRIC
    ] = pd.to_numeric(
        model_data[
            TARGET_METRIC
        ],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    model_data = (
        model_data
        .dropna(
            subset=[
                "query_id",
                TARGET_METRIC
            ]
        )
    )


    print(
        f"Valid queries: "
        f"{len(model_data):,}"
    )


    data[
        model_name
    ] = model_data


# ============================================================
# VERIFY QUERY ALIGNMENT
# ============================================================

print("\n")
print("=" * 75)
print(
    "VERIFYING QUERY ALIGNMENT"
)
print("=" * 75)


baseline = data[
    "Visual Baseline"
]


for model_name in [
    "Rare-Aware",
    "Metadata-Aware",
    "Combined Metadata + Rare"
]:

    model_df = data[
        model_name
    ]


    baseline_ids = set(
        baseline["query_id"]
    )

    model_ids = set(
        model_df["query_id"]
    )


    common_ids = (
        baseline_ids &
        model_ids
    )


    print(
        f"{model_name}: "
        f"{len(common_ids):,} "
        f"common queries"
    )


# ============================================================
# RUN PAIRED WILCOXON TESTS
# ============================================================

print("\n")
print("=" * 75)
print(
    "RUNNING WILCOXON TESTS"
)
print("=" * 75)


comparisons = [

    (
        "Metadata-Aware",
        "Visual Baseline"
    ),

    (
        "Rare-Aware",
        "Visual Baseline"
    ),

    (
        "Combined Metadata + Rare",
        "Visual Baseline"
    )
]


all_results = []


for model_name, baseline_name in comparisons:

    print("\n")
    print("-" * 75)

    print(
        f"Comparison:"
    )

    print(
        f"  {model_name}"
    )

    print(
        f"  vs"
    )

    print(
        f"  {baseline_name}"
    )

    print("-" * 75)


    model_df = data[
        model_name
    ]

    baseline_df = data[
        baseline_name
    ]


    # --------------------------------------------------------
    # Merge using query ID
    # --------------------------------------------------------

    merged = pd.merge(

        baseline_df,

        model_df,

        on="query_id",

        how="inner",

        suffixes=(
            "_baseline",
            "_model"
        )
    )


    # --------------------------------------------------------
    # Sort for deterministic pairing
    # --------------------------------------------------------

    merged = merged.sort_values(
        "query_id"
    ).reset_index(
        drop=True
    )


    baseline_values = (
        merged[
            f"{TARGET_METRIC}_baseline"
        ]
        .to_numpy(
            dtype=float
        )
    )


    model_values = (
        merged[
            f"{TARGET_METRIC}_model"
        ]
        .to_numpy(
            dtype=float
        )
    )


    n = len(
        merged
    )


    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    baseline_mean = (
        np.mean(
            baseline_values
        )
    )

    model_mean = (
        np.mean(
            model_values
        )
    )


    difference = (
        model_mean -
        baseline_mean
    )


    median_difference = (
        np.median(
            model_values -
            baseline_values
        )
    )


    # --------------------------------------------------------
    # Wilcoxon
    # --------------------------------------------------------

    differences = (
        model_values -
        baseline_values
    )


    # If every difference is zero,
    # Wilcoxon cannot be calculated normally.

    if np.all(
        differences == 0
    ):

        statistic = 0.0
        p_value = 1.0

    else:

        test_result = wilcoxon(

            model_values,

            baseline_values,

            alternative="two-sided",

            zero_method="wilcox",

            method="auto"
        )


        statistic = (
            test_result.statistic
        )

        p_value = (
            test_result.pvalue
        )


    significant = (
        p_value < ALPHA
    )


    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    improvement_pp = (
        difference * 100
    )


    relative_improvement = (

        difference /
        baseline_mean
    ) * 100


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Paired queries: "
        f"{n:,}"
    )

    print(
        f"Baseline mean NDCG@10: "
        f"{baseline_mean * 100:.4f}%"
    )

    print(
        f"{model_name} mean NDCG@10: "
        f"{model_mean * 100:.4f}%"
    )

    print(
        f"Mean improvement: "
        f"{improvement_pp:+.4f} pp"
    )

    print(
        f"Relative improvement: "
        f"{relative_improvement:+.4f}%"
    )

    print(
        f"Median per-query difference: "
        f"{median_difference * 100:+.4f} pp"
    )

    print(
        f"Wilcoxon statistic: "
        f"{statistic:.4f}"
    )

    print(
        f"p-value: "
        f"{p_value:.10f}"
    )


    if significant:

        print(
            "Result: "
            "STATISTICALLY SIGNIFICANT "
            f"(p < {ALPHA})"
        )

    else:

        print(
            "Result: "
            "NOT statistically significant "
            f"(p >= {ALPHA})"
        )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    all_results.append({

        "comparison":
            f"{model_name} vs {baseline_name}",

        "model":
            model_name,

        "baseline":
            baseline_name,

        "metric":
            "NDCG@10",

        "n_paired_queries":
            n,

        "baseline_mean":
            baseline_mean * 100,

        "model_mean":
            model_mean * 100,

        "improvement_pp":
            improvement_pp,

        "relative_improvement_percent":
            relative_improvement,

        "median_per_query_difference_pp":
            median_difference * 100,

        "wilcoxon_statistic":
            statistic,

        "p_value":
            p_value,

        "alpha":
            ALPHA,

        "significant":
            significant
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print(
    "WILCOXON RESULTS SUMMARY"
)
print("=" * 75)


summary_columns = [

    "comparison",

    "n_paired_queries",

    "baseline_mean",

    "model_mean",

    "improvement_pp",

    "wilcoxon_statistic",

    "p_value",

    "significant"
]


summary = (
    results_df[
        summary_columns
    ]
    .copy()
)


for column in [
    "baseline_mean",
    "model_mean",
    "improvement_pp",
    "wilcoxon_statistic",
    "p_value"
]:

    summary[column] = (
        summary[column]
        .round(6)
    )


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# PAPER-READY INTERPRETATION
# ============================================================

print("\n")
print("=" * 75)
print(
    "PAPER-READY INTERPRETATION"
)
print("=" * 75)


for _, row in (
    results_df.iterrows()
):

    comparison = (
        row["comparison"]
    )

    improvement = (
        row["improvement_pp"]
    )

    p = (
        row["p_value"]
    )


    if p < 0.001:

        significance_text = (
            "p < 0.001"
        )

    else:

        significance_text = (
            f"p = {p:.4f}"
        )


    print(
        f"\n{comparison}:"
    )

    print(
        f"NDCG@10 improvement = "
        f"{improvement:+.2f} percentage points; "
        f"{significance_text}."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print(
    "WILCOXON ANALYSIS COMPLETE"
)
print("=" * 75)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)