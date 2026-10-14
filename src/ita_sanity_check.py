import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V4 ITA SKIN-APPEARANCE SANITY CHECK
# ============================================================
#
# IMPORTANT:
# ITA is an estimated appearance proxy.
# It is NOT a ground-truth skin-tone label.
#
# Method:
#   1. Locate lesion-centered peri-lesional annulus
#   2. Remove obvious border / illumination artifacts
#   3. Remove very dark pixels likely to be hair / lesion
#   4. Remove highly saturated / strongly red pixels
#   5. Convert remaining pixels to CIELAB
#   6. Require positive, stable b* values
#   7. Robust percentile filtering
#   8. Median L* and b*
#   9. Calculate ITA
#  10. Report diagnostics
#
# Formula:
# ITA = arctan((L* - 50) / b*) * 180 / pi
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Swayam\Desktop\RS PROJ"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

SPLIT_FILE = os.path.join(
    DATA_DIR,
    "splits",
    "test.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "ita_sanity_samples_v4.csv"
)

CONTACT_SHEET = os.path.join(
    RESULTS_DIR,
    "ita_sanity_contact_sheet_v4.jpg"
)

# Number of images for manual sanity checking
N_SAMPLE = 50

RANDOM_SEED = 42


# ============================================================
# ANNULUS PARAMETERS
# ============================================================

# HAM10000 images are generally lesion-centered.
# We sample an outer peri-lesional ring around the center.

INNER_RADIUS = 0.20
OUTER_RADIUS = 0.45


# ============================================================
# IMAGE FILTERING PARAMETERS
# ============================================================

# RGB brightness limits
MIN_RGB = 20
MAX_RGB = 245

# HSV saturation upper limit
MAX_SATURATION = 0.80

# HSV value limits
MIN_VALUE = 30
MAX_VALUE = 245

# LAB filtering
#
# b* is particularly important because ITA divides by b*.
# Very small / negative b* values can produce unstable
# extreme positive or negative ITA estimates.
MIN_B = 8.0

# Reasonable L* range for skin appearance.
MIN_L = 15.0
MAX_L = 95.0


# ============================================================
# ROBUST PERCENTILE FILTERING
# ============================================================

LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90

# Minimum valid pixels
MIN_VALID_PIXELS = 1000

# Minimum fraction of annulus pixels that must survive
MIN_VALID_FRACTION = 0.05


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("V4 ITA SKIN-APPEARANCE SANITY CHECK")
print("=" * 70)

print()
print("IMPORTANT:")
print(
    "ITA is being used as an estimated appearance proxy."
)
print(
    "It is NOT a ground-truth skin-tone label."
)

print()
print("Method:")
print("Outer peri-lesional annulus")
print("→ artifact filtering")
print("→ CIELAB conversion")
print("→ positive/stable b* filtering")
print("→ robust percentile filtering")
print("→ median L* and b*")
print("→ ITA calculation")

print()
print(f"Test file: {SPLIT_FILE}")
print(f"Sample size: {N_SAMPLE}")
print(f"Random seed: {RANDOM_SEED}")


# ============================================================
# LOAD TEST DATA
# ============================================================

if not os.path.exists(SPLIT_FILE):

    raise FileNotFoundError(
        f"Test split not found:\n{SPLIT_FILE}"
    )

test_df = pd.read_csv(
    SPLIT_FILE
)

# ------------------------------------------------------------
# Detect image ID column
# ------------------------------------------------------------

possible_id_columns = [
    "image_id",
    "isic_id",
    "image",
    "id"
]

IMAGE_ID_COLUMN = None

for column in possible_id_columns:

    if column in test_df.columns:

        IMAGE_ID_COLUMN = column
        break


if IMAGE_ID_COLUMN is None:

    raise ValueError(
        "Could not find image ID column.\n"
        f"Available columns: {list(test_df.columns)}"
    )


# ------------------------------------------------------------
# Detect diagnosis column
# ------------------------------------------------------------

possible_class_columns = [
    "dx",
    "diagnosis",
    "class",
    "label"
]

CLASS_COLUMN = None

for column in possible_class_columns:

    if column in test_df.columns:

        CLASS_COLUMN = column
        break


if CLASS_COLUMN is None:

    raise ValueError(
        "Could not find diagnosis/class column.\n"
        f"Available columns: {list(test_df.columns)}"
    )


print()
print(f"Test images: {len(test_df):,}")
print(
    f"Image ID column: {IMAGE_ID_COLUMN}"
)
print(
    f"Class column: {CLASS_COLUMN}"
)


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

    files = glob.glob(
        pattern,
        recursive=True
    )

    for filepath in files:

        filename = os.path.basename(
            filepath
        )

        image_id = os.path.splitext(
            filename
        )[0]

        image_lookup[
            image_id
        ] = filepath


print(
    f"Images indexed: {len(image_lookup):,}"
)


# ============================================================
# VERIFY TEST IMAGES
# ============================================================

test_image_ids = (
    test_df[
        IMAGE_ID_COLUMN
    ]
    .astype(str)
    .tolist()
)

missing_images = [
    image_id
    for image_id in test_image_ids
    if image_id not in image_lookup
]

print(
    f"Images located: "
    f"{len(test_image_ids) - len(missing_images):,}"
)

if len(missing_images) > 0:

    print()
    print(
        f"⚠ Missing images: "
        f"{len(missing_images)}"
    )

    print(
        missing_images[:20]
    )

else:

    print(
        "✓ All test images located."
    )


# ============================================================
# IMAGE PROCESSING FUNCTION
# ============================================================

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
# RANDOM SAMPLE
# ============================================================

available_test_df = test_df[
    test_df[
        IMAGE_ID_COLUMN
    ].astype(str).isin(
        image_lookup.keys()
    )
].copy()


if len(
    available_test_df
) < N_SAMPLE:

    sample_df = available_test_df.copy()

else:

    sample_df = (
        available_test_df
        .sample(
            n=N_SAMPLE,
            random_state=RANDOM_SEED
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CALCULATE ITA
# ============================================================

print()
print("=" * 70)
print("CALCULATING V4 ITA ESTIMATES")
print("=" * 70)

results = []

images_for_contact_sheet = []


for idx, row in sample_df.iterrows():

    image_id = str(
        row[
            IMAGE_ID_COLUMN
        ]
    )

    dx = row[
        CLASS_COLUMN
    ]

    filepath = image_lookup[
        image_id
    ]


    image = cv2.imread(
        filepath
    )


    result = calculate_ita_v4(
        image
    )


    record = {

        "image_id":
            image_id,

        "dx":
            dx,

        "ita":
            result["ita"],

        "median_L":
            result["median_L"],

        "median_b":
            result["median_b"],

        "valid_skin_pixels":
            result[
                "valid_skin_pixels"
            ],

        "annulus_pixels":
            result[
                "annulus_pixels"
            ],

        "valid_fraction":
            result[
                "valid_fraction"
            ],

        "status":
            result[
                "status"
            ],

        "filepath":
            filepath
    }


    results.append(
        record
    )


    images_for_contact_sheet.append(
        (
            image_id,
            dx,
            result["ita"],
            image
        )
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# PRINT SAMPLE RESULTS
# ============================================================

print()
print("=" * 70)
print("V4 ITA SAMPLE RESULTS")
print("=" * 70)

display_columns = [
    "image_id",
    "dx",
    "ita",
    "median_L",
    "median_b",
    "valid_skin_pixels",
    "valid_fraction",
    "status"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

valid_results = results_df[
    np.isfinite(
        results_df[
            "ita"
        ]
    )
].copy()


print()
print("=" * 70)
print("V4 ITA SUMMARY")
print("=" * 70)


if len(valid_results) > 0:

    print(
        valid_results[
            "ita"
        ].describe().to_string()
    )

else:

    print(
        "No valid ITA estimates."
    )


print()
print(
    f"Valid estimates: "
    f"{len(valid_results)} / {len(results_df)}"
)

print(
    f"Missing estimates: "
    f"{len(results_df) - len(valid_results)}"
)


# ============================================================
# EXTREME VALUE CHECK
# ============================================================

if len(valid_results) > 0:

    extreme = valid_results[
        (
            valid_results["ita"] < -70
        )
        |
        (
            valid_results["ita"] > 85
        )
    ]


    print()
    print("=" * 70)
    print("EXTREME ITA CHECK")
    print("=" * 70)

    print(
        f"Extreme estimates: "
        f"{len(extreme)}"
    )

    if len(extreme) > 0:

        print()

        print(
            extreme[
                [
                    "image_id",
                    "dx",
                    "ita",
                    "median_L",
                    "median_b",
                    "valid_skin_pixels"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "✓ No extreme ITA estimates."
        )


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print(
    f"✓ Sample CSV saved:"
)

print(
    OUTPUT_CSV
)


# ============================================================
# CONTACT SHEET
# ============================================================

print()
print(
    "Creating contact sheet..."
)


n_images = len(
    images_for_contact_sheet
)

cols = 5

rows = int(
    np.ceil(
        n_images / cols
    )
)


fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(
        15,
        rows * 3.2
    )
)


axes = np.array(
    axes
).reshape(
    rows,
    cols
)


for i, (
    image_id,
    dx,
    ita,
    image
) in enumerate(
    images_for_contact_sheet
):

    r = i // cols
    c = i % cols

    ax = axes[
        r,
        c
    ]


    if image is not None:

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(
            image_rgb
        )


    if np.isfinite(ita):

        ita_text = (
            f"ITA={ita:.1f}"
        )

    else:

        ita_text = "ITA=NA"


    ax.set_title(
        f"{image_id}\n"
        f"dx={dx} | {ita_text}",
        fontsize=8
    )

    ax.axis(
        "off"
    )


# Hide unused axes

for i in range(
    n_images,
    rows * cols
):

    r = i // cols
    c = i % cols

    axes[
        r,
        c
    ].axis(
        "off"
    )


plt.tight_layout()

plt.savefig(
    CONTACT_SHEET,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    f"✓ Contact sheet saved:"
)

print(
    CONTACT_SHEET
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("V4 ITA SANITY CHECK COMPLETE")
print("=" * 70)