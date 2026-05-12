#!/usr/bin/env python3
"""Estimate FSD50K pair-collision probability before and after curriculum filters.

A collision means the sampled pair is not a clean separation pair because it is
the same source/file, or both FSD50K sources share at least one known label.
The source-level analysis can also measure collision risk after CLAP-similarity
thresholding, and after the curriculum sampler rejects source/file/label overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent

BROAD_FSD_LABELS = {
    "Animal",
    "Domestic_animals_and_pets",
    "Domestic_sounds_and_home_sounds",
    "Human_group_actions",
    "Human_voice",
    "Liquid",
    "Motor_vehicle_(road)",
    "Music",
    "Musical_instrument",
    "Percussion",
    "Plucked_string_instrument",
    "Speech",
    "Vehicle",
    "Water",
    "Wild_animals",
    "Wind_instrument_and_woodwind_instrument",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def load_sources(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        if "data" in payload:
            return payload["data"]
        if "sources" in payload:
            return payload["sources"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported source manifest shape: {path}")


def load_fsd_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    by_wav: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            labels = [label for label in row.get("labels", "").split(",") if label]
            by_wav[f"{row['fname']}.wav"] = {
                "split": row.get("split", ""),
                "labels": labels,
                "specific_labels": [
                    label for label in labels if label not in BROAD_FSD_LABELS
                ],
            }
    return by_wav


def load_original_fsd_auto_caption_sources(
    dev_json: Path,
    eval_json: Path,
    dev_gt_csv: Path,
    eval_gt_csv: Path,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    specs = [
        ("fsd50k_dev", dev_json, load_fsd_ground_truth(dev_gt_csv)),
        ("fsd50k_eval", eval_json, load_fsd_ground_truth(eval_gt_csv)),
    ]

    for dataset, json_path, ground_truth in specs:
        rows = read_json(json_path)["data"]
        for row in rows:
            wav_name = Path(str(row["wav"])).name
            meta = ground_truth.get(wav_name, {"split": "", "labels": [], "specific_labels": []})
            sources.append(
                {
                    "source_id": f"{dataset}:{wav_name}",
                    "dataset": dataset,
                    "split": meta.get("split", ""),
                    "wav": row["wav"],
                    "wav_name": wav_name,
                    "caption": row.get("caption", ""),
                    "captions": [row.get("caption", "")],
                    "caption_count": 1,
                    "labels": meta.get("labels", []),
                    "specific_labels": meta.get("specific_labels", []),
                }
            )

    return sources


def is_fsd(source: dict[str, Any]) -> bool:
    return str(source.get("dataset", "")).startswith("fsd50k")


def source_key(source: dict[str, Any], fallback_index: int) -> str:
    return str(source.get("source_id") or source.get("id") or source.get("wav") or fallback_index)


def wav_key(source: dict[str, Any]) -> str:
    return str(source.get("wav_name") or Path(str(source.get("wav", ""))).name)


def label_set(source: dict[str, Any]) -> set[str]:
    labels = source.get("specific_labels") or source.get("labels") or []
    return {str(label) for label in labels if str(label)}


def percentile(values: np.ndarray, pct: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, pct))


def rate(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask))


def sample_distinct_pairs(n: int, pair_count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left = rng.integers(0, n, size=pair_count, dtype=np.int64)
    right = rng.integers(0, n - 1, size=pair_count, dtype=np.int64)
    right = right + (right >= left)
    return left, right


def label_overlap_mask(
    left: np.ndarray,
    right: np.ndarray,
    label_sets: list[set[str]],
) -> np.ndarray:
    return label_overlap_counts(left, right, label_sets) > 0


def label_overlap_counts(
    left: np.ndarray,
    right: np.ndarray,
    label_sets: list[set[str]],
) -> np.ndarray:
    return np.asarray(
        [len(label_sets[int(i)] & label_sets[int(j)]) for i, j in zip(left, right)],
        dtype=np.int16,
    )


def collision_masks(
    left: np.ndarray,
    right: np.ndarray,
    source_keys: list[str],
    wav_keys: list[str],
    label_sets: list[set[str]],
) -> dict[str, np.ndarray]:
    same_source = np.asarray(
        [source_keys[int(i)] == source_keys[int(j)] for i, j in zip(left, right)],
        dtype=bool,
    )
    same_wav = np.asarray(
        [wav_keys[int(i)] == wav_keys[int(j)] for i, j in zip(left, right)],
        dtype=bool,
    )
    label_overlap_count = label_overlap_counts(left, right, label_sets)
    duplicate_source_or_wav = same_source | same_wav
    label_overlap = label_overlap_count > 0
    collision = duplicate_source_or_wav | label_overlap
    return {
        "same_source": same_source,
        "same_wav": same_wav,
        "duplicate_source_or_wav": duplicate_source_or_wav,
        "label_overlap_count": label_overlap_count,
        "label_overlap": label_overlap,
        "label_overlap_1": label_overlap_count == 1,
        "label_overlap_2": label_overlap_count == 2,
        "label_overlap_3plus": label_overlap_count >= 3,
        "collision": collision,
    }


def summarize_masked(
    name: str,
    mask: np.ndarray,
    collisions: dict[str, np.ndarray],
    similarities: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = int(np.sum(mask))
    summary: dict[str, Any] = {
        "name": name,
        "sampled_pairs": int(mask.size),
        "selected_pairs": selected,
        "selected_fraction": rate(mask),
    }
    if selected == 0:
        summary.update(
            {
                "same_source_rate": None,
                "same_wav_rate": None,
                "duplicate_source_or_wav_rate": None,
                "label_overlap_rate": None,
                "label_overlap_1_rate": None,
                "label_overlap_2_rate": None,
                "label_overlap_3plus_rate": None,
                "mean_overlapping_labels_when_any": None,
                "collision_rate": None,
            }
        )
        return summary

    selected_label_counts = collisions["label_overlap_count"][mask]
    overlapping_label_counts = selected_label_counts[selected_label_counts > 0]
    summary.update(
        {
            "same_source_rate": rate(collisions["same_source"][mask]),
            "same_wav_rate": rate(collisions["same_wav"][mask]),
            "duplicate_source_or_wav_rate": rate(
                collisions["duplicate_source_or_wav"][mask]
            ),
            "label_overlap_rate": rate(collisions["label_overlap"][mask]),
            "label_overlap_1_rate": rate(collisions["label_overlap_1"][mask]),
            "label_overlap_2_rate": rate(collisions["label_overlap_2"][mask]),
            "label_overlap_3plus_rate": rate(collisions["label_overlap_3plus"][mask]),
            "mean_overlapping_labels_when_any": (
                float(np.mean(overlapping_label_counts))
                if overlapping_label_counts.size > 0
                else 0.0
            ),
            "collision_rate": rate(collisions["collision"][mask]),
        }
    )
    if similarities is not None:
        sims = similarities[mask]
        summary["similarity"] = {
            "mean": float(np.mean(sims)),
            "p05": percentile(sims, 5),
            "p25": percentile(sims, 25),
            "p50": percentile(sims, 50),
            "p75": percentile(sims, 75),
            "p95": percentile(sims, 95),
        }
    return summary


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.3f}%"


def format_pct6(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.6f}%"


def exact_duplicate_key_probability(keys: list[str]) -> dict[str, Any]:
    """Exact P(two distinct flat rows share the same key)."""
    counts = Counter(keys)
    total = len(keys)
    duplicate_ordered_pairs = sum(count * (count - 1) for count in counts.values())
    total_ordered_pairs = total * (total - 1)
    probability = (
        duplicate_ordered_pairs / total_ordered_pairs if total_ordered_pairs > 0 else 0.0
    )
    return {
        "rows": total,
        "unique_keys": len(counts),
        "max_rows_per_key": max(counts.values()) if counts else 0,
        "duplicate_ordered_pairs": duplicate_ordered_pairs,
        "total_ordered_pairs": total_ordered_pairs,
        "probability": probability,
    }


def duplicate_probability_report(
    name: str,
    source_keys: list[str],
    wav_keys: list[str],
    batch_size: int,
) -> dict[str, Any]:
    source_report = exact_duplicate_key_probability(source_keys)
    wav_report = exact_duplicate_key_probability(wav_keys)
    same_source_pair_prob = source_report["probability"]
    same_wav_pair_prob = wav_report["probability"]
    # SegmentMixer uses adjacent items inside a shuffled batch. Treat adjacent
    # pair events as a close approximation for "at least one bad neighbor".
    approx_batch_source_prob = 1.0 - (1.0 - same_source_pair_prob) ** batch_size
    approx_batch_wav_prob = 1.0 - (1.0 - same_wav_pair_prob) ** batch_size
    return {
        "name": name,
        "batch_size": batch_size,
        "same_source_pair_probability": same_source_pair_prob,
        "same_wav_pair_probability": same_wav_pair_prob,
        "approx_at_least_one_same_source_neighbor_in_batch": approx_batch_source_prob,
        "approx_at_least_one_same_wav_neighbor_in_batch": approx_batch_wav_prob,
        "source_key_counts": source_report,
        "wav_key_counts": wav_report,
    }


def exact_label_overlap_report(name: str, label_sets: list[set[str]]) -> dict[str, Any]:
    """Exact P(two distinct rows have 1, 2, or >=3 shared labels)."""
    counts = Counter(frozenset(labels) for labels in label_sets)
    groups = list(counts.items())
    total = len(label_sets)
    total_ordered_pairs = total * (total - 1)
    buckets = Counter()

    for left_idx, (left_labels, left_count) in enumerate(groups):
        if not left_labels:
            continue
        for right_idx, (right_labels, right_count) in enumerate(groups):
            overlap_count = len(left_labels & right_labels)
            if overlap_count == 0:
                continue
            if left_idx == right_idx:
                contribution = left_count * (left_count - 1)
            else:
                contribution = left_count * right_count
            if contribution <= 0:
                continue
            buckets[min(overlap_count, 3)] += contribution

    any_overlap = buckets[1] + buckets[2] + buckets[3]
    denominator = total_ordered_pairs if total_ordered_pairs > 0 else 1
    return {
        "name": name,
        "rows": total,
        "unique_label_sets": len(groups),
        "one_label_overlap_probability": buckets[1] / denominator,
        "two_label_overlap_probability": buckets[2] / denominator,
        "three_plus_label_overlap_probability": buckets[3] / denominator,
        "any_label_overlap_probability": any_overlap / denominator,
    }


def print_exact_duplicate_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(
        "| case | rows | unique sources | max rows/source | same-source pair | "
        "same-wav pair | approx same-source per batch |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        source_counts = row["source_key_counts"]
        print(
            "| {name} | {rows:,} | {unique:,} | {max_count:,} | {same_source} | "
            "{same_wav} | {batch_source} |".format(
                name=row["name"],
                rows=source_counts["rows"],
                unique=source_counts["unique_keys"],
                max_count=source_counts["max_rows_per_key"],
                same_source=format_pct6(row["same_source_pair_probability"]),
                same_wav=format_pct6(row["same_wav_pair_probability"]),
                batch_source=format_pct6(
                    row["approx_at_least_one_same_source_neighbor_in_batch"]
                ),
            )
        )


def print_exact_label_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("| case | rows | unique label sets | 1 label | 2 labels | >=3 labels | any label |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {name} | {rows:,} | {unique:,} | {one} | {two} | {three} | {any_label} |".format(
                name=row["name"],
                rows=row["rows"],
                unique=row["unique_label_sets"],
                one=format_pct6(row["one_label_overlap_probability"]),
                two=format_pct6(row["two_label_overlap_probability"]),
                three=format_pct6(row["three_plus_label_overlap_probability"]),
                any_label=format_pct6(row["any_label_overlap_probability"]),
            )
        )


def print_summary_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(
        "| case | selected | selected % | same source | same wav | same src/wav | "
        "1 label | 2 labels | >=3 labels | any label | any collision |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {name} | {selected_pairs:,} | {selected_fraction} | "
            "{same_source} | {same_wav} | {same_src_wav} | {label1} | {label2} | "
            "{label3plus} | {any_label} | {collision} |".format(
                name=row["name"],
                selected_pairs=row["selected_pairs"],
                selected_fraction=format_pct(row["selected_fraction"]),
                same_source=format_pct(row["same_source_rate"]),
                same_wav=format_pct(row["same_wav_rate"]),
                same_src_wav=format_pct(row["duplicate_source_or_wav_rate"]),
                label1=format_pct(row["label_overlap_1_rate"]),
                label2=format_pct(row["label_overlap_2_rate"]),
                label3plus=format_pct(row["label_overlap_3plus_rate"]),
                any_label=format_pct(row["label_overlap_rate"]),
                collision=format_pct(row["collision_rate"]),
            )
        )


def expand_flat_caption_rows(sources: list[dict[str, Any]]) -> tuple[list[str], list[str], list[set[str]]]:
    flat_source_keys: list[str] = []
    flat_wav_keys: list[str] = []
    flat_label_sets: list[set[str]] = []
    for index, source in enumerate(sources):
        captions = source.get("captions") or [source.get("caption", "")]
        count = max(len(captions), 1)
        src_key = source_key(source, index)
        wav = wav_key(source)
        labels = label_set(source)
        for _ in range(count):
            flat_source_keys.append(src_key)
            flat_wav_keys.append(wav)
            flat_label_sets.append(labels)
    return flat_source_keys, flat_wav_keys, flat_label_sets


def stage_expected_collision(
    stage_name: str,
    weights: dict[str, float],
    band_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return {"name": stage_name, "similarity_only_collision_rate": None}
    expected: dict[str, float] = {
        "duplicate_source_or_wav_rate": 0.0,
        "label_overlap_1_rate": 0.0,
        "label_overlap_2_rate": 0.0,
        "label_overlap_3plus_rate": 0.0,
        "label_overlap_rate": 0.0,
        "collision_rate": 0.0,
    }
    for band, weight in weights.items():
        row = band_rows[band]
        normalized_weight = weight / total_weight
        for key in expected:
            expected[key] += normalized_weight * (row[key] or 0.0)
    return {
        "name": stage_name,
        "similarity_only": expected,
        "with_similarity_and_overlap_rejection": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-fsd-auto-captions",
        action="store_true",
        help="Analyze the non-augmented FSD50K auto-caption JSONs instead of a curriculum manifest.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=WORKSPACE_DIR / "similarity_analysis/curriculum_upload_primary/sources.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=WORKSPACE_DIR
        / "similarity_analysis/curriculum_upload_primary/source_primary_embeddings.npy",
    )
    parser.add_argument(
        "--original-fsd-dev-json",
        type=Path,
        default=WORKSPACE_DIR / "Task_provide_dataset/fsd50k_dev_auto_caption.json",
    )
    parser.add_argument(
        "--original-fsd-eval-json",
        type=Path,
        default=WORKSPACE_DIR / "Task_provide_dataset/fsd50k_eval_auto_caption.json",
    )
    parser.add_argument(
        "--fsd-dev-gt-csv",
        type=Path,
        default=WORKSPACE_DIR / "paraphrase/FSD50K.ground_truth/dev.csv",
    )
    parser.add_argument(
        "--fsd-eval-gt-csv",
        type=Path,
        default=WORKSPACE_DIR / "paraphrase/FSD50K.ground_truth/eval.csv",
    )
    parser.add_argument("--pair-samples", type=int, default=500_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for approximate same-source neighbor risk in SegmentMixer.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=SCRIPT_DIR / "datafiles/curriculum/fsd50k_pair_collision_report.json",
    )
    args = parser.parse_args()

    if args.original_fsd_auto_captions:
        sources = load_original_fsd_auto_caption_sources(
            dev_json=args.original_fsd_dev_json,
            eval_json=args.original_fsd_eval_json,
            dev_gt_csv=args.fsd_dev_gt_csv,
            eval_gt_csv=args.fsd_eval_gt_csv,
        )
        embeddings = None
        input_paths = {
            "original_fsd_dev_json": str(args.original_fsd_dev_json),
            "original_fsd_eval_json": str(args.original_fsd_eval_json),
            "fsd_dev_gt_csv": str(args.fsd_dev_gt_csv),
            "fsd_eval_gt_csv": str(args.fsd_eval_gt_csv),
        }
        embedding_note = "not used; this run reports label/source collision without CLAP thresholds"
    else:
        sources_all = load_sources(args.sources)
        embeddings_all = np.load(args.embeddings).astype("float32")
        if embeddings_all.shape[0] != len(sources_all):
            raise ValueError(
                f"Embedding/source mismatch: {embeddings_all.shape[0]} embeddings for "
                f"{len(sources_all)} sources"
            )

        fsd_indices = [idx for idx, source in enumerate(sources_all) if is_fsd(source)]
        sources = [sources_all[idx] for idx in fsd_indices]
        embeddings = embeddings_all[fsd_indices]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        input_paths = {
            "sources": str(args.sources),
            "embeddings": str(args.embeddings),
        }
        embedding_note = "local cache uses one primary caption embedding per source"

    src_keys = [source_key(source, idx) for idx, source in enumerate(sources)]
    wavs = [wav_key(source) for source in sources]
    labels = [label_set(source) for source in sources]

    rng = np.random.default_rng(args.seed)

    source_left, source_right = sample_distinct_pairs(len(sources), args.pair_samples, rng)
    source_collisions = collision_masks(source_left, source_right, src_keys, wavs, labels)
    similarities = (
        np.einsum("ij,ij->i", embeddings[source_left], embeddings[source_right])
        if embeddings is not None
        else None
    )

    flat_src_keys, flat_wavs, flat_labels = expand_flat_caption_rows(sources)
    flat_left, flat_right = sample_distinct_pairs(len(flat_src_keys), args.pair_samples, rng)
    flat_collisions = collision_masks(flat_left, flat_right, flat_src_keys, flat_wavs, flat_labels)
    exact_duplicate_rows = [
        duplicate_probability_report(
            "flat caption-row pool",
            source_keys=flat_src_keys,
            wav_keys=flat_wavs,
            batch_size=args.batch_size,
        ),
        duplicate_probability_report(
            "source-level pool",
            source_keys=src_keys,
            wav_keys=wavs,
            batch_size=args.batch_size,
        ),
    ]
    exact_label_rows = [
        exact_label_overlap_report("flat caption-row pool", flat_labels),
        exact_label_overlap_report("source-level pool", labels),
    ]

    all_mask = np.ones(args.pair_samples, dtype=bool)
    before_rows = [
        summarize_masked("old flat caption-row random FSD50K pairs", all_mask, flat_collisions),
        summarize_masked("source-level random FSD50K pairs", all_mask, source_collisions, similarities),
    ]

    similarity_only_rows = []
    similarity_and_overlap_rows = []
    band_similarity_only_rows: dict[str, dict[str, Any]] = {}
    band_safe_rows = []
    stage_rows = []

    if similarities is not None:
        max_thresholds = [0.06, 0.125, 0.19, 0.29]
        for threshold in max_thresholds:
            sim_mask = similarities <= threshold
            similarity_only_rows.append(
                summarize_masked(
                    f"similarity only: sim <= {threshold}",
                    sim_mask,
                    source_collisions,
                    similarities,
                )
            )
            safe_mask = sim_mask & ~source_collisions["collision"]
            similarity_and_overlap_rows.append(
                summarize_masked(
                    f"sim <= {threshold} + reject same wav/source/labels",
                    safe_mask,
                    source_collisions,
                    similarities,
                )
            )

        band_specs = {
            "easy": (None, 0.06),
            "medium": (0.06, 0.19),
            "hard": (0.19, 0.29),
        }
        for name, (min_sim, max_sim) in band_specs.items():
            band_mask = similarities <= max_sim
            if min_sim is not None:
                band_mask &= similarities >= min_sim
            band_similarity_only_rows[name] = summarize_masked(
                f"{name} band similarity only",
                band_mask,
                source_collisions,
                similarities,
            )
            band_safe_rows.append(
                summarize_masked(
                    f"{name} band + reject same wav/source/labels",
                    band_mask & ~source_collisions["collision"],
                    source_collisions,
                    similarities,
                )
            )

        stage_rows = [
            stage_expected_collision(
                "full_easy weights",
                {"easy": 0.85, "medium": 0.15},
                band_similarity_only_rows,
            ),
            stage_expected_collision(
                "full_medium weights",
                {"easy": 0.60, "medium": 0.30, "hard": 0.10},
                band_similarity_only_rows,
            ),
            stage_expected_collision(
                "full_hard weights",
                {"easy": 0.45, "medium": 0.35, "hard": 0.20},
                band_similarity_only_rows,
            ),
            stage_expected_collision(
                "full_hard_upper weights",
                {"easy": 0.35, "medium": 0.40, "hard": 0.25},
                band_similarity_only_rows,
            ),
        ]

    report = {
        "inputs": {
            **input_paths,
            "original_fsd_auto_captions": args.original_fsd_auto_captions,
            "embedding_note": embedding_note,
            "pair_samples": args.pair_samples,
            "seed": args.seed,
        },
        "fsd_sources": len(sources),
        "flat_caption_rows": len(flat_src_keys),
        "exact_duplicate_source_probability": exact_duplicate_rows,
        "exact_label_overlap_probability": exact_label_rows,
        "before": before_rows,
        "similarity_only_max_thresholds": similarity_only_rows,
        "similarity_and_overlap_rejection_max_thresholds": similarity_and_overlap_rows,
        "band_similarity_only": list(band_similarity_only_rows.values()),
        "band_similarity_and_overlap_rejection": band_safe_rows,
        "stage_expected_collision": stage_rows,
        "collision_definition": (
            "same source OR same wav OR at least one overlapping specific_labels/labels"
        ),
    }
    write_json(args.out_json, report)

    print(f"FSD50K sources: {len(sources):,}")
    print(f"Flat caption rows expanded from those sources: {len(flat_src_keys):,}")
    print(f"Random pair samples per estimate: {args.pair_samples:,}")
    print_exact_duplicate_table(
        "Exact same-source/same-wav collision from duplicate caption rows",
        exact_duplicate_rows,
    )
    print_exact_label_table(
        "Exact label-overlap collision probability",
        exact_label_rows,
    )
    print_summary_table("Before curriculum filtering", before_rows)
    if similarities is not None:
        print_summary_table("After CLAP similarity threshold only", similarity_only_rows)
        print_summary_table(
            "After CLAP similarity + hard overlap rejection",
            similarity_and_overlap_rows,
        )
        print_summary_table("Curriculum bands", list(band_similarity_only_rows.values()) + band_safe_rows)

        print("\nExpected collision from current full-finetune band weights")
        print(
            "| stage | same src/wav | 1 label | 2 labels | >=3 labels | "
            "any label | any collision | after rejection |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in stage_rows:
            similarity_only = row["similarity_only"]
            print(
                f"| {row['name']} | "
                f"{format_pct(similarity_only['duplicate_source_or_wav_rate'])} | "
                f"{format_pct(similarity_only['label_overlap_1_rate'])} | "
                f"{format_pct(similarity_only['label_overlap_2_rate'])} | "
                f"{format_pct(similarity_only['label_overlap_3plus_rate'])} | "
                f"{format_pct(similarity_only['label_overlap_rate'])} | "
                f"{format_pct(similarity_only['collision_rate'])} | "
                f"{format_pct(row['with_similarity_and_overlap_rejection'])} |"
            )
    else:
        print("\nCLAP threshold sections skipped because no original-caption embeddings were provided.")
        print("This original-dataset run measures source/file/label collision probability only.")

    print(f"\nWrote JSON report: {args.out_json}")


if __name__ == "__main__":
    main()
