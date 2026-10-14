"""Non-destructive correction of saved HAM10000 retrieval experiments.

Run from project root: python -m backend.evaluation.audit
Requires numpy (already a backend dependency); does not load/train a model.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .metrics import correct_legacy_ndcg, retrieval_metrics

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("precision_at_5", "precision_at_10", "ndcg_at_10", "rr_at_10")
SOURCES = {
    "visual": "baseline_metrics_per_query.csv",
    "metadata": "final_metadata_query_results.csv",
    "rarity": "final_rare_aware_query_results.csv",
    "combined": "final_combined_query_results.csv",
}
RAW = {"visual": "baseline_retrieval_results.csv", "rarity": SOURCES["rarity"]}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def indexed(rows, key="image_id"):
    result = {}
    for row in rows:
        identity = row.get(key, "").strip()
        if not identity or identity in result:
            raise ValueError(f"Missing/duplicate {key}: {identity}")
        result[identity] = row
    return result


def audit_splits(root):
    splits = {name: read_csv(root / "data/splits" / f"{name}.csv")
              for name in ("train", "val", "test")}
    master = indexed(read_csv(root / "data/HAM10000_metadata.csv"))
    ids, lesions = {}, {}
    lesion_labels = {}
    for name, rows in splits.items():
        ids[name] = set(indexed(rows))
        lesions[name] = set()
        for row in rows:
            if not row.get("lesion_id") or not row.get("dx"):
                raise ValueError(f"Missing lesion/diagnosis in {name}")
            if row["image_id"] not in master or any(
                row.get(field) != master[row["image_id"]].get(field)
                for field in ("lesion_id", "dx", "dx_type", "age", "sex", "localization")
            ):
                raise ValueError(f"Split metadata mismatch: {row['image_id']}")
            lesion = row["lesion_id"]
            if lesion in lesion_labels and lesion_labels[lesion] != row["dx"]:
                raise ValueError(f"Conflicting diagnoses in lesion {lesion}")
            lesion_labels[lesion] = row["dx"]
            lesions[name].add(lesion)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[left] & ids[right] or lesions[left] & lesions[right]:
            raise ValueError(f"Image/lesion leakage: {left}/{right}")
    if set.union(*ids.values()) != set(master):
        raise ValueError("Splits do not cover exactly the master metadata")
    reference = read_csv(root / "results/train_embedding_metadata.csv")
    if reference != splits["train"]:
        raise ValueError("Gallery metadata differs from train split/order")
    info = {name: {"images": len(rows), "lesions": len(lesions[name]),
                   "classes": dict(Counter(row["dx"] for row in rows))}
            for name, rows in splits.items()}
    info.update({"image_and_lesion_overlap": 0,
                 "patient_overlap": "unknown: patient identifiers unavailable",
                 "pixel_duplicate_check": "not performed",
                 "index_embedding_alignment": "not verified by this saved-results audit"})
    return splits, indexed(reference), info


def query_index(rows):
    return indexed([{**row, "image_id": row.get("query_image_id", row.get("image_id"))}
                    for row in rows])


def correct_method(method, root, queries, gallery):
    saved = query_index(read_csv(root / "results" / SOURCES[method]))
    if set(saved) != set(queries):
        raise ValueError(f"{method}: incomplete or unexpected query coverage")
    raw = query_index(read_csv(root / "results" / RAW[method])) if method in RAW else None
    if raw is not None and set(raw) != set(queries):
        raise ValueError(f"{method}: raw query coverage mismatch")
    counts = Counter(row["dx"] for row in gallery.values())
    output = []
    for image_id, query in queries.items():
        row = saved[image_id]
        if row["query_dx"] != query["dx"]:
            raise ValueError(f"{method}: query label mismatch {image_id}")
        old = float(row["ndcg_at_10"])
        p5, p10 = float(row["precision_at_5"]), float(row["precision_at_10"])
        rr = float(row.get("mrr", row.get("reciprocal_rank")))
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (p5, p10, rr)):
            raise ValueError(f"{method}: invalid saved metric")
        if abs(p5 * 5 - round(p5 * 5)) > 1e-7 or not 0 <= p10 * 10 - p5 * 5 <= 5 + 1e-7:
            raise ValueError(f"{method}: inconsistent P@5/P@10")
        if (p10 == 0) != (rr == 0) or (rr and not any(abs(rr - 1 / r) < 1e-7 for r in range(1, 11))):
            raise ValueError(f"{method}: invalid truncated RR")
        corrected = correct_legacy_ndcg(old, p10, counts[query["dx"]])
        metrics = dict(zip(METRICS, (p5, p10, corrected, rr)))
        if raw is not None:
            ranking = raw[image_id]
            retrieved = ranking["retrieved_image_ids"].split("|")
            labels = ranking["retrieved_dx"].split("|")
            if ranking["query_dx"] != query["dx"] or len(retrieved) != 10 or len(set(retrieved)) != 10 or len(labels) != 10:
                raise ValueError(f"{method}: invalid top-10 ranking")
            for candidate_id, label in zip(retrieved, labels):
                if candidate_id not in gallery or gallery[candidate_id]["dx"] != label:
                    raise ValueError(f"{method}: candidate not in gallery/label mismatch")
                if gallery[candidate_id]["lesion_id"] == query["lesion_id"]:
                    raise ValueError(f"{method}: same-lesion retrieval")
            recomputed = retrieval_metrics([label == query["dx"] for label in labels], counts[query["dx"]])
            if any(abs(metrics[key] - recomputed[key]) > 1e-7 for key in METRICS):
                raise ValueError(f"{method}: saved metrics disagree with raw ranking")
            metrics = recomputed
        output.append({"method": method, "image_id": image_id, "lesion_id": query["lesion_id"],
                       "dx": query["dx"], "legacy_ndcg_at_10": old,
                       "evidence": "raw_ranking" if raw is not None else "algebraic_metric_recovery",
                       **metrics})
    return output


def summarize(method_rows, repetitions=2000, seed=42):
    """Paired, class-stratified lesion-cluster percentile bootstrap.

    Resample lesions with replacement, retaining all their images. Point estimates
    remain image-weighted within each class; macro gives equal weight to classes.
    The SAME cluster draws are used for every method, preserving paired comparisons.
    """
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    methods = list(method_rows)
    anchor = method_rows[methods[0]]
    for rows in method_rows.values():
        if [(r["image_id"], r["lesion_id"], r["dx"]) for r in rows] != [
            (r["image_id"], r["lesion_id"], r["dx"]) for r in anchor
        ]:
            raise ValueError("Methods are not query-aligned")
    values = np.array([[[row[key] for key in METRICS] for row in method_rows[m]] for m in methods])
    rng = np.random.default_rng(seed)
    stats, boots = {}, {}
    total_sums = np.zeros((repetitions, len(methods), len(METRICS)))
    total_counts = np.zeros(repetitions)
    for diagnosis in sorted({row["dx"] for row in anchor}):
        groups = defaultdict(list)
        for position, row in enumerate(anchor):
            if row["dx"] == diagnosis:
                groups[row["lesion_id"]].append(position)
        positions = [p for group in groups.values() for p in group]
        sums = np.array([values[:, group, :].sum(axis=1) for group in groups.values()])
        counts = np.array([len(group) for group in groups.values()])
        # Bounded chunks keep memory independent of large replicate requests.
        boot_sums = np.empty((repetitions, len(methods), len(METRICS)))
        boot_counts = np.empty(repetitions)
        for start in range(0, repetitions, 100):
            stop = min(start + 100, repetitions)
            draws = rng.integers(0, len(groups), size=(stop - start, len(groups)))
            boot_sums[start:stop] = sums[draws].sum(axis=1)
            boot_counts[start:stop] = counts[draws].sum(axis=1)
        boots[diagnosis] = boot_sums / boot_counts[:, None, None]
        stats[diagnosis] = (values[:, positions, :].mean(axis=1), len(positions), len(groups))
        total_sums += boot_sums
        total_counts += boot_counts
    classes = list(stats)
    stats["overall"] = (values.mean(axis=1), len(anchor), len({r["lesion_id"] for r in anchor}))
    stats["macro"] = (np.mean([stats[c][0] for c in classes], axis=0), *stats["overall"][1:])
    boots["overall"] = total_sums / total_counts[:, None, None]
    boots["macro"] = np.mean([boots[c] for c in classes], axis=0)
    summary, deltas = [], []
    for scope, (points, n_images, n_lesions) in stats.items():
        for mi, method in enumerate(methods):
            for ki, metric in enumerate(METRICS):
                low, high = np.quantile(boots[scope][:, mi, ki], [0.025, 0.975])
                summary.append(dict(method=method, scope=scope, metric=metric,
                                    value=float(points[mi, ki]), ci_low=float(low), ci_high=float(high),
                                    images=n_images, lesions=n_lesions,
                                    small_sample=n_lesions < 30))
                if mi:
                    delta = boots[scope][:, mi, ki] - boots[scope][:, 0, ki]
                    low, high = np.quantile(delta, [0.025, 0.975])
                    deltas.append(dict(method=method, comparator=methods[0], scope=scope, metric=metric,
                                       delta=float(points[mi, ki] - points[0, ki]),
                                       ci_low=float(low), ci_high=float(high)))
    return summary, deltas


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    splits, gallery, split_info = audit_splits(ROOT)
    queries = indexed(splits["test"])
    corrected = {method: correct_method(method, ROOT, queries, gallery) for method in SOURCES}
    summary, deltas = summarize(corrected, args.bootstrap, args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "backend/audits" / f"evaluation-v2-{timestamp}"
    inputs = list((ROOT / "data/splits").glob("*.csv")) + [ROOT / "data/HAM10000_metadata.csv"]
    inputs += [ROOT / "results" / name for name in set(SOURCES.values()) | set(RAW.values()) | {"train_embedding_metadata.csv"}]
    inputs += list(Path(__file__).parent.glob("*.py"))
    # Preserve the exact source snapshots underlying the historical correction.
    inputs += list((ROOT / "src").glob("*.py"))
    manifest = {
        "schema_version": 2, "created_utc": timestamp,
        "purpose": "retrospective saved-results metric correction; NOT fresh inference or clinical validation",
        "test_status": "exploratory: historical test-set sweeps exist; not an untouched confirmatory holdout",
        "parameter_provenance": "Historical result files do not carry immutable run configs; exact generating settings unverified. Rare-only source currently specifies boost 0.02; deployed combined uses 0.03.",
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed, "unit": "lesion", "strata": "diagnosis", "paired": True},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "splits": split_info,
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))},
    }
    output.mkdir(parents=True, exist_ok=False)
    for method, rows in corrected.items():
        write_csv(output / f"{method}_per_query.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_deltas.csv", deltas)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Corrected retrospective retrieval evaluation", "",
             "Research only. No model training, parameter selection, deployment, or fresh inference was performed.", "",
             "All four historical methods cover 1,502 test images. Metadata/gallery checks passed; see manifest for counts and hashes.",
             "Baseline and rarity: recomputed from saved ranked IDs and checked against recovered metrics.",
             "Metadata and combined: DCG recovered algebraically from legacy NDCG and P@10; ranked IDs were not saved, so candidate provenance cannot be verified for those runs.", "",
             "NDCG uses all same-diagnosis gallery images in the ideal denominator. MRR is explicitly truncated at 10.",
             "95% percentile intervals use paired, diagnosis-stratified lesion-cluster bootstrap. No multiplicity-adjusted hypothesis testing is claimed.", "",
             "| Method | P@5 | P@10 | Corrected NDCG@10 (95% CI) | MRR@10 |", "|---|---:|---:|---:|---:|"]
    for method in SOURCES:
        entries = {r["metric"]: r for r in summary if r["scope"] == "overall" and r["method"] == method}
        ndcg = entries["ndcg_at_10"]
        lines.append(f"| {method} | {entries['precision_at_5']['value']:.4f} | {entries['precision_at_10']['value']:.4f} | {ndcg['value']:.4f} ({ndcg['ci_low']:.4f}-{ndcg['ci_high']:.4f}) | {entries['rr_at_10']['value']:.4f} |")
    lines += ["", "## Limitations and next gates", "",
              "- Existing test-set sweeps mean this is exploratory evidence, not a new independent test.",
              "- Lesion separation is checked. Patient separation and pixel/near-duplicate separation are not established.",
              "- Confidence intervals condition on the fixed gallery; they do not account for training variability or source/site shift.",
              "- Rare classes have small numbers of independent lesions; see summary.csv small_sample flags.",
              "- No calibrated malignancy probability, diagnostic sensitivity, or clinical safety claim follows from these retrieval metrics.",
              "- Original results, legacy statistics/plots, weights, dataset, index and UI remain unchanged; old NDCG-derived claims are superseded, not silently overwritten.",
              "- Next: validation-only model development, record ranked IDs/config/model hashes, then independently sourced external testing and clinician evaluation."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
