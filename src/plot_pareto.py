import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\Swayam\Desktop\RS PROJ"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "results",
    "pareto_rarity_results.csv"
)

OUTPUT_PLOT = os.path.join(
    BASE_DIR,
    "results",
    "pareto_rarity_curve.png"
)

df = pd.read_csv(INPUT_CSV)

plt.figure(figsize=(10, 7))

plt.plot(
    df["ndcg_at_10"],
    df["macro_ndcg_at_10"],
    marker="o"
)

for _, row in df.iterrows():

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

plt.xlabel("Overall NDCG@10 (%)")
plt.ylabel("Macro NDCG@10 (%)")

plt.title(
    "Rarity-Aware Retrieval: "
    "Overall vs Macro NDCG@10"
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

print("\nPlot saved to:")
print(OUTPUT_PLOT)