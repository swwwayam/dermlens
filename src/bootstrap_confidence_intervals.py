import os
import numpy as np
import pandas as pd


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
    "bootstrap_confidence_intervals.csv"
)

N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


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
# METRICS
# ============================================================

METRICS = [
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_10",
    "mrr"
]


# ============================================================
# METRIC COLUMN ALIASES
# ============================================================

# Some older result files use "reciprocal_rank"
# while newer files use "mrr".

METRIC_ALIASES = {

    "precision_at_5": [
        "precision_at_5"
    ],

    "precision_at_10": [
        "precision_at_10"
    ],

    "ndcg_at_10": [
        "ndcg_at_10"
    ],

    "mrr": [
        "mrr",
        "reciprocal_rank"
    ]
}


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


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

    raise ValueError(
        "Could not identify diagnosis/class "
        "column.\nAvailable columns:\n"
        + str(list(df.columns))
    )


def find_metric_column(
    df,
    metric
):

    possible_columns = (
        METRIC_ALIASES[metric]
    )

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


def bootstrap_ci(
    values,
    n_bootstrap=1000,
    confidence=0.95
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    # --------------------------------------------------------
    # Point estimate
    # --------------------------------------------------------

    point_estimate = np.mean(
        values
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float
    )

    for i in range(
        n_bootstrap
    ):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    # --------------------------------------------------------
    # Percentile confidence interval
    # --------------------------------------------------------

    alpha = (
        1.0 -
        confidence
    )

    lower = np.percentile(
        bootstrap_means,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (
            1 -
            alpha / 2
        )
    )

    return (
        point_estimate,
        lower,
        upper
    )


def get_sample_flag(
    class_name,
    n_queries
):

    # Explicitly flag the minority classes
    # requested for the paper.

    if class_name in [
        "df",
        "vasc",
        "akiec"
    ]:

        if n_queries <= 25:

            return "VERY SMALL SAMPLE"

        return "SMALL SAMPLE"

    if n_queries < 100:

        return "SMALL SAMPLE"

    return ""


# ============================================================
# START
# ============================================================

print("=" * 75)
print(
    "BOOTSTRAP 95% CONFIDENCE INTERVALS"
)
print("=" * 75)

print(
    f"\nBootstrap iterations: "
    f"{N_BOOTSTRAP}"
)

print(
    f"Confidence level: "
    f"{CONFIDENCE_LEVEL * 100:.0f}%"
)

print(
    f"Random seed: "
    f"{RANDOM_SEED}"
)


# ============================================================
# LOAD MODELS
# ============================================================

all_results = []

for model_name, filename in (
    MODEL_FILES.items()
):

    filepath = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(
        filepath
    ):

        print(
            f"\n⚠ SKIPPING: "
            f"{model_name}"
        )

        print(
            f"File not found: "
            f"{filepath}"
        )

        continue


    print("\n")
    print("=" * 75)
    print(
        f"PROCESSING: {model_name}"
    )
    print("=" * 75)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        filepath
    )

    print(
        f"Queries loaded: "
        f"{len(df):,}"
    )

    print(
        "Columns:"
    )

    print(
        list(df.columns)
    )


    # --------------------------------------------------------
    # Find class column
    # --------------------------------------------------------

    class_column = (
        find_class_column(df)
    )

    print(
        f"\nClass column: "
        f"{class_column}"
    )


    # --------------------------------------------------------
    # Find metric columns
    # --------------------------------------------------------

    metric_columns = {}

    print(
        "\nMetric mapping:"
    )

    for metric in METRICS:

        column = find_metric_column(
            df,
            metric
        )

        if column is None:

            print(
                f"  ⚠ {metric}: NOT FOUND"
            )

        else:

            metric_columns[
                metric
            ] = column

            print(
                f"  ✓ {metric} "
                f"→ {column}"
            )


    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = sorted(
        df[class_column]
        .dropna()
        .unique()
    )

    print(
        f"\nClasses found: "
        f"{len(classes)}"
    )


    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for cls in classes:

        class_df = df[
            df[class_column] == cls
        ].copy()

        n_queries = len(
            class_df
        )

        sample_flag = (
            get_sample_flag(
                cls,
                n_queries
            )
        )


        print("\n" + "-" * 65)

        print(
            f"Class: {cls}"
        )

        print(
            f"Queries: {n_queries}"
        )

        if sample_flag:

            print(
                f"⚠ {sample_flag}"
            )


        # ----------------------------------------------------
        # Each metric
        # ----------------------------------------------------

        for metric in METRICS:

            if metric not in (
                metric_columns
            ):

                continue


            actual_column = (
                metric_columns[
                    metric
                ]
            )


            values = (
                class_df[
                    actual_column
                ]
                .astype(float)
                .to_numpy()
            )


            point, lower, upper = (
                bootstrap_ci(
                    values,
                    N_BOOTSTRAP,
                    CONFIDENCE_LEVEL
                )
            )


            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            all_results.append({

                "model":
                    model_name,

                "class":
                    cls,

                "n_queries":
                    n_queries,

                "sample_flag":
                    sample_flag,

                "metric":
                    metric,

                "source_column":
                    actual_column,

                "estimate":
                    point * 100,

                "ci_lower":
                    lower * 100,

                "ci_upper":
                    upper * 100,

                "ci_width":
                    (
                        upper -
                        lower
                    ) * 100
            })


            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"{metric:18s}"
                f"{point * 100:7.2f}% "
                f"["
                f"{lower * 100:7.2f}, "
                f"{upper * 100:7.2f}"
                f"]"
            )


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
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 75)
print(
    "BOOTSTRAP RESULTS"
)
print("=" * 75)

if len(results_df) > 0:

    display_columns = [

        "model",
        "class",
        "n_queries",
        "sample_flag",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper"
    ]

    display_df = (
        results_df[
            display_columns
        ].copy()
    )

    for column in [
        "estimate",
        "ci_lower",
        "ci_upper"
    ]:

        display_df[column] = (
            display_df[column]
            .round(2)
        )

    print(
        display_df.to_string(
            index=False
        )
    )

else:

    print(
        "No results were generated."
    )


# ============================================================
# SMALL-SAMPLE SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print(
    "MINORITY / SMALL-SAMPLE CLASSES"
)
print("=" * 75)


if len(results_df) > 0:

    small_classes = (
        results_df[
            results_df[
                "sample_flag"
            ] != ""
        ]
        [
            [
                "model",
                "class",
                "n_queries",
                "sample_flag"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "class",
                "model"
            ]
        )
    )


    if len(
        small_classes
    ) > 0:

        print(
            small_classes.to_string(
                index=False
            )
        )

    else:

        print(
            "No small-sample classes."
        )


# ============================================================
# EXPLICIT PAPER WARNING
# ============================================================

print("\n")
print("=" * 75)
print(
    "IMPORTANT INTERPRETATION NOTE"
)
print("=" * 75)

print(
    """
The classes df, vasc, and akiec have relatively
small numbers of test queries. Their bootstrap
confidence intervals are therefore expected to be
wider and should be interpreted cautiously.

The confidence intervals quantify uncertainty in
the reported per-class retrieval metrics; they do
not remove the limitation caused by the small
number of minority-class queries.
"""
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print(
    "BOOTSTRAP COMPLETE"
)
print("=" * 75)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)