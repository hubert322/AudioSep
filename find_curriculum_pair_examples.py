#!/usr/bin/env python3
"""Find non-collision curriculum pair examples across similarity ranges."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_ANALYSIS_DIR = SCRIPT_DIR.parent.parent / "similarity_analysis" / "curriculum_upload_primary"

DEFAULT_RANGES = [
    ("easy", None, 0.06),
    ("medium", 0.06, 0.19),
    ("hard", 0.19, 0.29),
    ("sim_0.30_0.50", 0.30, 0.50),
    ("sim_0.50_0.70", 0.50, 0.70),
    ("sim_gt_0.70", 0.70, 1.01),
]


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
    raise ValueError(f"Unsupported source manifest format: {path}")


def default_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def is_fsd(source: dict[str, Any]) -> bool:
    return str(source.get("dataset", "")).startswith("fsd50k")


def source_id(source: dict[str, Any], idx: int) -> str:
    return str(source.get("source_id") or source.get("id") or source.get("wav") or idx)


def wav_name(source: dict[str, Any]) -> str:
    return str(source.get("wav_name") or Path(str(source.get("wav", ""))).name)


def label_set(source: dict[str, Any]) -> set[str]:
    return {str(label) for label in source.get("specific_labels") or source.get("labels") or []}


def has_collision(left: dict[str, Any], right: dict[str, Any], left_idx: int, right_idx: int) -> bool:
    if left_idx == right_idx:
        return True
    if source_id(left, left_idx) == source_id(right, right_idx):
        return True
    if wav_name(left) == wav_name(right):
        return True

    left_labels = label_set(left)
    right_labels = label_set(right)
    if left_labels and right_labels and left_labels & right_labels:
        return True
    return False


def clean_caption(source: dict[str, Any]) -> str:
    captions = source.get("captions") or []
    if captions:
        return str(captions[0])
    return str(source.get("caption", ""))


def format_labels(source: dict[str, Any]) -> str:
    labels = sorted(label_set(source))
    return ", ".join(labels) if labels else "(no labels)"


def in_range(similarity: float, min_similarity: float | None, max_similarity: float) -> bool:
    if similarity >= max_similarity:
        return False
    if min_similarity is not None and similarity < min_similarity:
        return False
    return True


def example_row(
    range_name: str,
    similarity: float,
    left_idx: int,
    right_idx: int,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    left = sources[left_idx]
    right = sources[right_idx]
    return {
        "range": range_name,
        "similarity": similarity,
        "left": {
            "source_index": int(left_idx),
            "source_id": source_id(left, left_idx),
            "dataset": left.get("dataset", ""),
            "wav": left.get("wav", ""),
            "caption": clean_caption(left),
            "labels": sorted(label_set(left)),
        },
        "right": {
            "source_index": int(right_idx),
            "source_id": source_id(right, right_idx),
            "dataset": right.get("dataset", ""),
            "wav": right.get("wav", ""),
            "caption": clean_caption(right),
            "labels": sorted(label_set(right)),
        },
        "shared_labels": sorted(label_set(left) & label_set(right)),
    }


def write_markdown(path: Path, examples: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Curriculum Pair Examples", ""]
    for range_name, rows in examples.items():
        lines.extend([f"## {range_name}", ""])
        if not rows:
            lines.extend(["No safe examples found.", ""])
            continue
        for idx, row in enumerate(rows, start=1):
            left = row["left"]
            right = row["right"]
            lines.extend(
                [
                    f"{idx}. similarity: `{row['similarity']:.4f}`",
                    f"   - A: {left['caption']}",
                    f"     - dataset/wav: `{left['dataset']}` / `{Path(left['wav']).name}`",
                    f"     - labels: {', '.join(left['labels']) if left['labels'] else '(no labels)'}",
                    f"   - B: {right['caption']}",
                    f"     - dataset/wav: `{right['dataset']}` / `{Path(right['wav']).name}`",
                    f"     - labels: {', '.join(right['labels']) if right['labels'] else '(no labels)'}",
                    "   - shared labels: none",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def find_examples(
    sources: list[dict[str, Any]],
    embeddings: np.ndarray,
    examples_per_range: int,
    seed: int,
    max_random_pairs: int,
    batch_pairs: int,
    fsd_only: bool,
) -> dict[str, list[dict[str, Any]]]:
    if fsd_only:
        keep_indices = [idx for idx, source in enumerate(sources) if is_fsd(source)]
    else:
        keep_indices = list(range(len(sources)))
    if len(keep_indices) < 2:
        raise ValueError("Need at least two candidate sources.")

    rng = np.random.default_rng(seed)
    order_rng = random.Random(seed)
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in DEFAULT_RANGES}
    seen_pairs: set[tuple[int, int]] = set()
    candidates = np.asarray(keep_indices, dtype=np.int64)
    sampled = 0

    while sampled < max_random_pairs and any(
        len(rows) < examples_per_range for rows in examples.values()
    ):
        current_batch = min(batch_pairs, max_random_pairs - sampled)
        sampled += current_batch
        left_positions = rng.integers(0, len(candidates), size=current_batch, dtype=np.int64)
        right_positions = rng.integers(0, len(candidates) - 1, size=current_batch, dtype=np.int64)
        right_positions = right_positions + (right_positions >= left_positions)
        left_indices = candidates[left_positions]
        right_indices = candidates[right_positions]
        similarities = np.einsum("ij,ij->i", embeddings[left_indices], embeddings[right_indices])

        row_order = list(range(current_batch))
        order_rng.shuffle(row_order)
        for row_idx in row_order:
            left_idx = int(left_indices[row_idx])
            right_idx = int(right_indices[row_idx])
            pair_key = tuple(sorted((left_idx, right_idx)))
            if pair_key in seen_pairs:
                continue
            left = sources[left_idx]
            right = sources[right_idx]
            if has_collision(left, right, left_idx, right_idx):
                continue

            similarity = float(similarities[row_idx])
            for range_name, min_similarity, max_similarity in DEFAULT_RANGES:
                if len(examples[range_name]) >= examples_per_range:
                    continue
                if not in_range(similarity, min_similarity, max_similarity):
                    continue
                examples[range_name].append(
                    example_row(range_name, similarity, left_idx, right_idx, sources)
                )
                seen_pairs.add(pair_key)
                break

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=default_path(
            SCRIPT_DIR / "datafiles/curriculum/sources.json",
            DEFAULT_LOCAL_ANALYSIS_DIR / "sources.json",
        ),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=default_path(
            SCRIPT_DIR / "datafiles/curriculum/source_embeddings.npy",
            DEFAULT_LOCAL_ANALYSIS_DIR / "source_primary_embeddings.npy",
        ),
    )
    parser.add_argument("--examples-per-range", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-random-pairs", type=int, default=5_000_000)
    parser.add_argument("--batch-pairs", type=int, default=200_000)
    parser.add_argument("--include-non-fsd", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=SCRIPT_DIR / "datafiles/curriculum/curriculum_pair_examples.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=SCRIPT_DIR / "datafiles/curriculum/curriculum_pair_examples.md",
    )
    args = parser.parse_args()

    sources = load_sources(args.sources)
    embeddings = np.load(args.embeddings).astype("float32")
    if embeddings.shape[0] != len(sources):
        raise ValueError(
            f"Embedding/source count mismatch: {embeddings.shape[0]} embeddings for "
            f"{len(sources)} sources"
        )
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    examples = find_examples(
        sources=sources,
        embeddings=embeddings,
        examples_per_range=args.examples_per_range,
        seed=args.seed,
        max_random_pairs=args.max_random_pairs,
        batch_pairs=args.batch_pairs,
        fsd_only=not args.include_non_fsd,
    )

    report = {
        "sources": str(args.sources),
        "embeddings": str(args.embeddings),
        "examples_per_range": args.examples_per_range,
        "fsd_only": not args.include_non_fsd,
        "ranges": [
            {"name": name, "min_similarity": min_similarity, "max_similarity": max_similarity}
            for name, min_similarity, max_similarity in DEFAULT_RANGES
        ],
        "examples": examples,
    }
    write_json(args.out_json, report)
    write_markdown(args.out_md, examples)

    for range_name, rows in examples.items():
        print(f"\n{range_name}: {len(rows)} examples")
        for row in rows:
            print(
                f"  sim={row['similarity']:.4f} | "
                f"{row['left']['caption']} || {row['right']['caption']}"
            )
            print(f"    labels A: {', '.join(row['left']['labels'])}")
            print(f"    labels B: {', '.join(row['right']['labels'])}")
    print(f"\nWrote JSON: {args.out_json}")
    print(f"Wrote Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
