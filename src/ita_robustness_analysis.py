import os
import glob
import cv2
import numpy as np
import pandas as pd


# ============================================================
# FULL ITA ROBUSTNESS ANALYSIS — V4 LOCKED METHOD
# ============================================================
#
# IMPORTANT:
# ITA is an estimated appearance proxy.
# It is NOT a ground-truth skin-tone label.
#
# This script reuses the EXACT calculate_ita_v4() implementation
# from ita_sanity_check.py.
#
# Pipeline:
#   V4 ITA on all 1,502 test images
#        ↓
#   invalid estimates kept as NaN
#        ↓
#   valid ITA estimates ranked into 3 tertiles
#        ↓
#   Visual Baseline / Metadata-Aware / Rare-Aware
#        ↓
#   P@5 / P@10 / NDCG@10 / MRR
#        ↓
#   1,000-query bootstrap 95% CIs
#
# No skin-tone ground truth is claimed.
# Tertiles are reported as low/middle/high ITA appearance proxy.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Swayam\Desktop\RS PROJ"

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SPLIT_FILE = os.path.join(
    DATA_DIR,
    "splits",
    "test.csv"
)

ITA_OUTPUT = os.path.join(
    RESULTS_DIR,
    "ita_full_test_v4.csv"
)

TERTILE_OUTPUT = os.path.join(
    RESULTS_DIR,
    "ita_tertile_assignments.csv"
)

METRICS_OUTPUT = os.path.join(
    RESULTS_DIR,
    "ita_robustness_metrics.csv"
)

CI_OUTPUT = os.path.join(
    RESULTS_DIR,
    "ita_robustness_bootstrap_cis.csv"
)

N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

METRICS = [
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_10",
    "mrr"
]

MODEL_FILES = {
    "Visual Baseline":
        "baseline_metrics_per_query.csv",

    "Metadata-Aware":
        "final_metadata_query_results.csv",

    "Rare-Aware":
        "final_test_results.csv"
}


# ============================================================
# EXACT V4 ITA FUNCTION
# ============================================================
# ============================================================
# LOCKED V4 ITA PARAMETERS
# ============================================================

# Peri-lesional annulus
INNER_RADIUS = 0.20
OUTER_RADIUS = 0.45

# RGB / HSV artifact filtering
MIN_RGB = 20
MAX_RGB = 245

MAX_SATURATION = 0.80

MIN_VALUE = 30
MAX_VALUE = 245

# CIELAB filtering
MIN_B = 8.0

MIN_L = 15.0
MAX_L = 95.0

# Robust percentile filtering
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90

# Validity requirements
MIN_VALID_PIXELS = 1000
MIN_VALID_FRACTION = 0.05


def calculate_ita_v4(
    image
):
    """
    Calculate robust ITA estimate from
    an outer peri-lesional annulus.
    """

    if image is None:

        return {
            "ita": np.nan,
            "median_L": np.nan,
            "median_b": np.nan,
            "valid_skin_pixels": 0,
            "annulus_pixels": 0,
            "valid_fraction": 0.0,
            "status": "invalid_image"
        }


    # --------------------------------------------------------
    # Resize for consistent processing
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (450, 450),
        interpolation=cv2.INTER_AREA
    )


    h, w = image.shape[:2]

    cy = h / 2.0
    cx = w / 2.0

    yy, xx = np.ogrid[
        :h,
        :w
    ]


    # --------------------------------------------------------
    # Radial distance
    # --------------------------------------------------------

    distance = np.sqrt(
        (
            (xx - cx) / w
        ) ** 2
        +
        (
            (yy - cy) / h
        ) ** 2
    )


    annulus_mask = (
        (distance >= INNER_RADIUS)
        &
        (distance <= OUTER_RADIUS)
    )


    annulus_pixels = int(
        np.sum(
            annulus_mask
        )
    )


    if annulus_pixels == 0:

        return {
            "ita": np.nan,
            "median_L": np.nan,
            "median_b": np.nan,
            "valid_skin_pixels": 0,
            "annulus_pixels": 0,
            "valid_fraction": 0.0,
            "status": "empty_annulus"
        }


    # ========================================================
    # RGB
    # ========================================================

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # HSV
    # ========================================================

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]


    # Normalize saturation
    S_norm = (
        S.astype(
            np.float32
        )
        / 255.0
    )


    # ========================================================
    # BASIC ARTIFACT FILTER
    # ========================================================

    rgb_min = np.min(
        rgb,
        axis=2
    )

    rgb_max = np.max(
        rgb,
        axis=2
    )


    basic_mask = (

        annulus_mask

        &

        (rgb_min >= MIN_RGB)

        &

        (rgb_max <= MAX_RGB)

        &

        (S_norm <= MAX_SATURATION)

        &

        (V >= MIN_VALUE)

        &

        (V <= MAX_VALUE)
    )


    # ========================================================
    # CIELAB
    # ========================================================

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )


    # OpenCV LAB:
    #
    # L: 0-255
    # a: 0-255, centered around 128
    # b: 0-255, centered around 128
    #
    # Convert to conventional CIELAB approximately.

    L = (
        lab[:, :, 0]
        .astype(np.float32)
        * 100.0
        / 255.0
    )

    a = (
        lab[:, :, 1]
        .astype(np.float32)
        - 128.0
    )

    b = (
        lab[:, :, 2]
        .astype(np.float32)
        - 128.0
    )


    # ========================================================
    # LAB SKIN-APPEARANCE FILTER
    # ========================================================

    lab_mask = (

        basic_mask

        &

        (L >= MIN_L)

        &

        (L <= MAX_L)

        # Stable positive b*
        &
        (b >= MIN_B)
    )


    # ========================================================
    # EXTRACT VALID PIXELS
    # ========================================================

    L_values = L[
        lab_mask
    ]

    b_values = b[
        lab_mask
    ]

    a_values = a[
        lab_mask
    ]


    valid_before_percentile = len(
        L_values
    )


    if valid_before_percentile < MIN_VALID_PIXELS:

        return {
            "ita": np.nan,
            "median_L": np.nan,
            "median_b": np.nan,
            "valid_skin_pixels":
                valid_before_percentile,
            "annulus_pixels":
                annulus_pixels,
            "valid_fraction":
                valid_before_percentile
                /
                max(
                    annulus_pixels,
                    1
                ),
            "status":
                "too_few_valid_pixels"
        }


    # ========================================================
    # ROBUST PERCENTILE FILTERING
    # ========================================================

    L_low, L_high = np.percentile(
        L_values,
        [
            LOW_PERCENTILE,
            HIGH_PERCENTILE
        ]
    )

    b_low, b_high = np.percentile(
        b_values,
        [
            LOW_PERCENTILE,
            HIGH_PERCENTILE
        ]
    )


    robust_mask = (

        lab_mask

        &

        (L >= L_low)

        &

        (L <= L_high)

        &

        (b >= b_low)

        &

        (b <= b_high)
    )


    L_final = L[
        robust_mask
    ]

    b_final = b[
        robust_mask
    ]


    valid_pixels = len(
        L_final
    )


    valid_fraction = (
        valid_pixels
        /
        max(
            annulus_pixels,
            1
        )
    )


    if valid_pixels < MIN_VALID_PIXELS:

        return {
            "ita": np.nan,
            "median_L": np.nan,
            "median_b": np.nan,
            "valid_skin_pixels":
                valid_pixels,
            "annulus_pixels":
                annulus_pixels,
            "valid_fraction":
                valid_fraction,
            "status":
                "too_few_robust_pixels"
        }


    if valid_fraction < MIN_VALID_FRACTION:

        return {
            "ita": np.nan,
            "median_L": np.nan,
            "median_b": np.nan,
            "valid_skin_pixels":
                valid_pixels,
            "annulus_pixels":
                annulus_pixels,
            "valid_fraction":
                valid_fraction,
            "status":
                "low_valid_fraction"
        }


    # ========================================================
    # ROBUST MEDIANS
    # ========================================================

    median_L = float(
        np.median(
            L_final
        )
    )

    median_b = float(
        np.median(
            b_final
        )
    )


    # ========================================================
    # FINAL ITA
    # ========================================================

    # b* is guaranteed positive by the filter.

    ita = (
        np.degrees(
            np.arctan(
                (
                    median_L - 50.0
                )
                /
                median_b
            )
        )
    )


    return {
        "ita": float(ita),
        "median_L": median_L,
        "median_b": median_b,
        "valid_skin_pixels":
            valid_pixels,
        "annulus_pixels":
            annulus_pixels,
        "valid_fraction":
            valid_fraction,
        "status":
            "valid"
    }



# ============================================================
# HELPERS
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def find_query_id_column(df):
    return find_column(
        df,
        [
            "query_image_id",
            "image_id",
            "query_id",
            "isic_id",
            "image"
        ],
        "query/image ID"
    )


def find_class_column(df):
    return find_column(
        df,
        [
            "query_dx",
            "dx",
            "class",
            "label",
            "diagnosis"
        ],
        "diagnosis/class"
    )


def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    point = float(np.mean(values))

    boot = np.empty(N_BOOTSTRAP, dtype=float)

    for i in range(N_BOOTSTRAP):
        sample = rng.choice(
            values,
            size=n,
            replace=True
        )
        boot[i] = np.mean(sample)

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = float(
        np.percentile(
            boot,
            100 * alpha / 2
        )
    )

    upper = float(
        np.percentile(
            boot,
            100 * (1 - alpha / 2)
        )
    )

    return point, lower, upper


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FULL V4 ITA SKIN-APPEARANCE ROBUSTNESS ANALYSIS — V5")
print("=" * 70)

print()
print("IMPORTANT:")
print("ITA is an estimated appearance proxy.")
print("It is NOT a ground-truth skin-tone label.")

print()
print("Locked method:")
print("Outer peri-lesional annulus")
print("→ artifact filtering")
print("→ CIELAB conversion")
print("→ positive/stable b* filtering")
print("→ robust percentile filtering")
print("→ median L* and b*")
print("→ ITA calculation")

print()
print(f"Test file: {SPLIT_FILE}")
print("Full test size: 1,502 expected")
print(f"Bootstrap iterations: {N_BOOTSTRAP}")
print(f"Random seed: {RANDOM_SEED}")


# ============================================================
# LOAD TEST DATA
# ============================================================

if not os.path.exists(SPLIT_FILE):
    raise FileNotFoundError(
        f"Test split not found:\n{SPLIT_FILE}"
    )

test_df = pd.read_csv(SPLIT_FILE)

IMAGE_ID_COLUMN = find_column(
    test_df,
    ["image_id", "isic_id", "image", "id"],
    "image ID"
)

CLASS_COLUMN = find_class_column(test_df)

test_df[IMAGE_ID_COLUMN] = (
    test_df[IMAGE_ID_COLUMN]
    .astype(str)
)

print()
print(f"Test queries: {len(test_df):,}")
print(f"Image ID column: {IMAGE_ID_COLUMN}")
print(f"Class column: {CLASS_COLUMN}")


# ============================================================
# BUILD IMAGE LOOKUP
# ============================================================

print()
print("=" * 70)
print("LOCATING TEST IMAGES")
print("=" * 70)

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

image_lookup = {}

print()
print("Scanning image directories...")

for extension in image_extensions:
    pattern = os.path.join(
        DATA_DIR,
        "**",
        extension
    )

    for filepath in glob.glob(
        pattern,
        recursive=True
    ):
        image_id = os.path.splitext(
            os.path.basename(filepath)
        )[0]

        image_lookup[image_id] = filepath

print(
    f"Images indexed: {len(image_lookup):,}"
)


# ============================================================
# VERIFY TEST IMAGES
# ============================================================

missing = [
    image_id
    for image_id in test_df[IMAGE_ID_COLUMN]
    if image_id not in image_lookup
]

if missing:
    print(
        f"⚠ Missing test images: {len(missing)}"
    )
    print(missing[:20])

    raise FileNotFoundError(
        f"{len(missing)} test images could not be located."
    )

print(
    f"✓ All {len(test_df):,} test images located."
)


# ============================================================
# LOAD OR CALCULATE V4 ITA
# ============================================================
#
# IMPORTANT:
# If ita_full_test_v4.csv already exists, DO NOT recalculate
# the 1,502 images. This makes the analysis restart-safe.
# ============================================================

print()
print("=" * 70)
print("LOADING / CALCULATING V4 ITA")
print("=" * 70)

if os.path.exists(ITA_OUTPUT):

    print()
    print("✓ Existing V4 ITA file found.")
    print("Loading cached ITA estimates instead of recalculating images.")

    ita_df = pd.read_csv(
        ITA_OUTPUT
    )

    required_ita_columns = [
        "image_id",
        "dx",
        "ita"
    ]

    missing_ita_columns = [
        column
        for column in required_ita_columns
        if column not in ita_df.columns
    ]

    if missing_ita_columns:
        raise ValueError(
            "Cached ITA file is missing required columns: "
            + str(missing_ita_columns)
        )

    ita_df["image_id"] = (
        ita_df["image_id"]
        .astype(str)
    )

    print(
        f"Cached ITA records loaded: "
        f"{len(ita_df):,}"
    )

else:

    print()
    print("No cached V4 ITA file found.")
    print("Calculating V4 ITA on the full test set...")

    ita_records = []

    for i, row in test_df.iterrows():

        image_id = str(row[IMAGE_ID_COLUMN])
        dx = row[CLASS_COLUMN]

        filepath = image_lookup[image_id]

        image = cv2.imread(filepath)

        if image is None:
            raise ValueError(
                f"Could not read image: {filepath}"
            )

        result = calculate_ita_v4(image)

        ita_records.append({
            "image_id": image_id,
            "dx": dx,
            "ita": result["ita"],
            "median_L": result["median_L"],
            "median_b": result["median_b"],
            "valid_skin_pixels": result["valid_skin_pixels"],
            "annulus_pixels": result["annulus_pixels"],
            "valid_fraction": result["valid_fraction"],
            "status": result["status"]
        })

        if (
            (i + 1) % 100 == 0
            or (i + 1) == len(test_df)
        ):
            print(
                f"Processed "
                f"{i + 1:,}/"
                f"{len(test_df):,}"
            )

    ita_df = pd.DataFrame(
        ita_records
    )

    ita_df.to_csv(
        ITA_OUTPUT,
        index=False
    )

    print()
    print(
        f"✓ V4 ITA estimates saved to:\n"
        f"{ITA_OUTPUT}"
    )


# ============================================================
# VERIFY ITA COVERAGE
# ============================================================

# Make sure the cached/calculated ITA file covers the same
# test set that we are evaluating.

test_ids = set(
    test_df[IMAGE_ID_COLUMN]
    .astype(str)
)

ita_ids = set(
    ita_df["image_id"]
    .astype(str)
)

missing_ita_records = (
    test_ids - ita_ids
)

extra_ita_records = (
    ita_ids - test_ids
)

if missing_ita_records:

    raise ValueError(
        f"ITA file is missing "
        f"{len(missing_ita_records)} test images."
    )

if extra_ita_records:

    print(
        f"⚠ ITA file contains "
        f"{len(extra_ita_records)} extra records."
    )

# Keep only the current test set and preserve test ordering.
ita_df["image_id"] = (
    ita_df["image_id"]
    .astype(str)
)

ita_df = (
    test_df[
        [IMAGE_ID_COLUMN, CLASS_COLUMN]
    ]
    .rename(
        columns={
            IMAGE_ID_COLUMN: "image_id",
            CLASS_COLUMN: "dx"
        }
    )
    .merge(
        ita_df,
        on="image_id",
        how="left",
        suffixes=(
            "_test",
            "_ita"
        )
    )
)

# Resolve duplicate dx columns if the cached file has dx.
if "dx_test" in ita_df.columns:

    ita_df["dx"] = ita_df["dx_test"]

    ita_df = ita_df.drop(
        columns=[
            column
            for column in [
                "dx_test",
                "dx_ita"
            ]
            if column in ita_df.columns
        ]
    )

print()
print("=" * 70)
print("FULL ITA SUMMARY")
print("=" * 70)

print(
    f"Total test images: "
    f"{len(ita_df):,}"
)

valid_ita = ita_df[
    np.isfinite(
        pd.to_numeric(
            ita_df["ita"],
            errors="coerce"
        )
    )
].copy()

invalid_ita = ita_df[
    ~np.isfinite(
        pd.to_numeric(
            ita_df["ita"],
            errors="coerce"
        )
    )
].copy()

print(
    f"Valid ITA estimates: "
    f"{len(valid_ita):,}"
)

print(
    f"Missing/invalid estimates: "
    f"{len(invalid_ita):,}"
)

print(
    f"Valid percentage: "
    f"{100 * len(valid_ita) / len(ita_df):.2f}%"
)

if len(valid_ita) > 0:

    print()

    print(
        valid_ita["ita"]
        .describe()
        .to_string()
    )

if "status" in ita_df.columns:

    print()
    print("Status counts:")

    print(
        ita_df["status"]
        .value_counts()
        .to_string()
    )


# ============================================================
# CREATE TERTILES
# ============================================================
#
# Rank first so the three groups are as equal in size as
# possible even if repeated ITA values occur.
#
# These are deliberately named:
#   low_ITA
#   middle_ITA
#   high_ITA
#
# NOT "dark/medium/light skin tone".
# ============================================================

print()
print("=" * 70)
print("CREATING ITA TERTILES")
print("=" * 70)

ita_df["ita_tertile"] = pd.NA

if len(valid_ita) < 3:
    raise ValueError(
        "Fewer than 3 valid ITA estimates; "
        "cannot create tertiles."
    )

ranked_indices = (
    valid_ita["ita"]
    .rank(
        method="first"
    )
)

tertiles = pd.qcut(
    ranked_indices,
    q=3,
    labels=[
        "low_ITA",
        "middle_ITA",
        "high_ITA"
    ]
)

ita_df.loc[
    valid_ita.index,
    "ita_tertile"
] = tertiles.astype(str).values

ita_df.to_csv(
    TERTILE_OUTPUT,
    index=False
)

print()
print(
    ita_df["ita_tertile"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("ITA ranges by tertile:")

print(
    ita_df[
        ita_df["ita_tertile"].notna()
    ]
    .groupby("ita_tertile", observed=True)["ita"]
    .agg(
        n="count",
        min="min",
        median="median",
        max="max"
    )
    .to_string()
)


# ============================================================
# LOAD RETRIEVAL RESULTS
# ============================================================

print()
print("=" * 70)
print("LOADING RETRIEVAL RESULTS")
print("=" * 70)

model_data = {}

for model_name, filename in MODEL_FILES.items():

    filepath = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Required result file missing:\n{filepath}"
        )

    print()
    print(f"Loading: {model_name}")
    print(f"File: {filename}")

    df = pd.read_csv(filepath)

    query_id_column = find_query_id_column(df)
    class_column = find_class_column(df)

    df[query_id_column] = (
        df[query_id_column]
        .astype(str)
    )

    print(
        f"Queries: {len(df):,}"
    )

    print(
        f"Query ID column: {query_id_column}"
    )

    print(
        f"Class column: {class_column}"
    )

    # --------------------------------------------------------
    # Baseline MRR naming
    # --------------------------------------------------------
    #
    # The visual baseline stores the per-query reciprocal
    # rank under "reciprocal_rank". This is exactly the
    # per-query contribution used to compute MRR:
    #
    #     MRR = mean(reciprocal_rank)
    #
    # Therefore we map it to the common "mrr" name used
    # throughout this analysis.
    # --------------------------------------------------------

    if (
        model_name == "Visual Baseline"
        and "mrr" not in df.columns
        and "reciprocal_rank" in df.columns
    ):
        df["mrr"] = df["reciprocal_rank"]

        print(
            "✓ Mapped baseline "
            "'reciprocal_rank' → 'mrr'"
        )

    missing_metrics = [
        metric
        for metric in METRICS
        if metric not in df.columns
    ]

    if missing_metrics:
        raise ValueError(
            f"{model_name} is missing metrics: "
            f"{missing_metrics}"
        )

    # Keep only columns needed for the analysis.
    keep = [
        query_id_column,
        class_column
    ] + METRICS

    model_df = df[keep].copy()

    model_df = model_df.rename(
        columns={
            query_id_column: "image_id",
            class_column: "dx"
        }
    )

    # Protect against duplicate query IDs.
    if model_df["image_id"].duplicated().any():
        duplicates = (
            model_df["image_id"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"{model_name} contains "
            f"{duplicates} duplicate query IDs."
        )

    model_data[model_name] = model_df


# ============================================================
# MERGE EACH MODEL WITH ITA TERTILES
# ============================================================

all_metrics = []

ita_join = ita_df[
    [
        "image_id",
        "ita",
        "ita_tertile"
    ]
].copy()

for model_name, model_df in model_data.items():

    merged = model_df.merge(
        ita_join,
        on="image_id",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(model_df):
        print()
        print(
            f"⚠ {model_name}: "
            f"{len(model_df) - len(merged)} queries "
            f"did not match ITA records."
        )

    merged["model"] = model_name

    all_metrics.append(
        merged
    )


metrics_df = pd.concat(
    all_metrics,
    ignore_index=True
)


# ============================================================
# VERIFY MODEL ALIGNMENT
# ============================================================

print()
print("=" * 70)
print("MODEL / ITA ALIGNMENT")
print("=" * 70)

for model_name in MODEL_FILES:

    subset = metrics_df[
        metrics_df["model"] == model_name
    ]

    valid = subset[
        subset["ita_tertile"].notna()
    ]

    print(
        f"{model_name}: "
        f"{len(subset):,} total queries | "
        f"{len(valid):,} valid ITA queries"
    )


# ============================================================
# COMPUTE POINT ESTIMATES BY TERTILE
# ============================================================

print()
print("=" * 70)
print("ITA TERTILE RETRIEVAL RESULTS")
print("=" * 70)

metric_records = []

for model_name in MODEL_FILES:

    model_df = metrics_df[
        metrics_df["model"] == model_name
    ].copy()

    for tertile in [
        "low_ITA",
        "middle_ITA",
        "high_ITA"
    ]:

        subset = model_df[
            model_df["ita_tertile"] == tertile
        ].copy()

        n_queries = len(subset)

        print()
        print(
            f"{model_name} | {tertile} | "
            f"n={n_queries}"
        )

        for metric in METRICS:

            values = pd.to_numeric(
                subset[metric],
                errors="coerce"
            ).to_numpy()

            values = values[
                np.isfinite(values)
            ]

            estimate = (
                np.mean(values)
                if len(values) > 0
                else np.nan
            )

            metric_records.append({
                "model": model_name,
                "ita_tertile": tertile,
                "n_queries": n_queries,
                "metric": metric,
                "estimate": estimate * 100
            })

            print(
                f"  {metric:18s}: "
                f"{estimate * 100:.2f}%"
            )


metrics_results_df = pd.DataFrame(
    metric_records
)

metrics_results_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


# ============================================================
# BOOTSTRAP 95% CIs
# ============================================================

print()
print("=" * 70)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 70)

ci_records = []

for model_name in MODEL_FILES:

    model_df = metrics_df[
        metrics_df["model"] == model_name
    ].copy()

    for tertile in [
        "low_ITA",
        "middle_ITA",
        "high_ITA"
    ]:

        subset = model_df[
            model_df["ita_tertile"] == tertile
        ].copy()

        n_queries = len(subset)

        for metric in METRICS:

            values = pd.to_numeric(
                subset[metric],
                errors="coerce"
            ).to_numpy()

            point, lower, upper = bootstrap_ci(
                values
            )

            if n_queries < 100:
                sample_flag = "SMALL SAMPLE"
            else:
                sample_flag = ""

            ci_records.append({
                "model": model_name,
                "ita_tertile": tertile,
                "n_queries": n_queries,
                "sample_flag": sample_flag,
                "metric": metric,
                "estimate": point * 100,
                "ci_lower": lower * 100,
                "ci_upper": upper * 100,
                "ci_width": (upper - lower) * 100
            })

            print(
                f"{model_name:22s} | "
                f"{tertile:12s} | "
                f"{metric:18s} | "
                f"{point * 100:6.2f}% "
                f"[{lower * 100:6.2f}, "
                f"{upper * 100:6.2f}]"
            )


ci_df = pd.DataFrame(
    ci_records
)

ci_df.to_csv(
    CI_OUTPUT,
    index=False
)


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

print()
print("=" * 70)
print("PAPER-READY ROBUSTNESS SUMMARY")
print("=" * 70)

if len(ci_df) > 0:

    summary_columns = [
        "model",
        "ita_tertile",
        "n_queries",
        "sample_flag",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper"
    ]

    print(
        ci_df[
            summary_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL OUTPUTS
# ============================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print()
print(f"Full ITA estimates:")
print(ITA_OUTPUT)

print()
print(f"Tertile assignments:")
print(TERTILE_OUTPUT)

print()
print(f"Point-estimate metrics:")
print(METRICS_OUTPUT)

print()
print(f"Bootstrap 95% CIs:")
print(CI_OUTPUT)

print()
print("=" * 70)
print("FULL ITA ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 70)