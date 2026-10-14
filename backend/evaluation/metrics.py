"""Binary retrieval metrics. Similarity is not a diagnostic probability."""

from math import isfinite, log2


def ideal_dcg(total_relevant: int, k: int = 10) -> float:
    if k < 1 or total_relevant < 0:
        raise ValueError("k must be positive and total_relevant nonnegative")
    return sum(1.0 / log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))


def retrieval_metrics(relevance, total_relevant: int, k: int = 10) -> dict:
    """Use the entire eligible gallery for IDCG, not the returned positives.

    Unreturned ranks count as non-relevant. RR is truncated at k.
    Relevance means matching the dataset diagnosis, not clinical equivalence.
    """
    values = list(relevance)[:k]
    if any(value not in (0, 1, False, True) for value in values):
        raise ValueError("Only binary relevance is supported")
    denominator = ideal_dcg(total_relevant, k)
    if sum(values) > total_relevant:
        raise ValueError("Returned positives exceed eligible gallery positives")
    dcg = sum(value / log2(rank + 1) for rank, value in enumerate(values, 1))
    return {
        "precision_at_5": sum(values[:5]) / 5,
        "precision_at_10": sum(values[:10]) / 10,
        "ndcg_at_10": dcg / denominator if denominator else 0.0,
        "rr_at_10": next((1.0 / rank for rank, value in enumerate(values, 1) if value), 0.0),
    } if k == 10 else {
        f"precision_at_{k}": sum(values) / k,
        f"ndcg_at_{k}": dcg / denominator if denominator else 0.0,
        f"rr_at_{k}": next((1.0 / rank for rank, value in enumerate(values, 1) if value), 0.0),
    }


def correct_legacy_ndcg(legacy_ndcg: float, precision: float,
                        total_relevant: int, k: int = 10) -> float:
    """Recover DCG from legacy NDCG and P@k without inventing ranked cases.

    Valid ONLY for the historical binary metric whose IDCG used the number of
    positives in the returned top-k. This is metric recovery, not fresh inference.
    """
    if not all(isfinite(value) and 0 <= value <= 1 for value in (legacy_ndcg, precision)):
        raise ValueError("Legacy NDCG and precision must be finite values in [0, 1]")
    count = round(precision * k)
    if abs(count - precision * k) > 1e-7 or count > total_relevant:
        raise ValueError("P@k is inconsistent with a binary ranking/gallery")
    if count == 0 and legacy_ndcg != 0:
        raise ValueError("Nonzero NDCG with zero retrieved positives")
    denominator = ideal_dcg(total_relevant, k)
    return legacy_ndcg * ideal_dcg(count, k) / denominator if denominator else 0.0
