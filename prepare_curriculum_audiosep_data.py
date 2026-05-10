#!/usr/bin/env python3
"""Build source-level curriculum data for AudioSep training.

This script converts the augmented FSD50K and Clotho caption files into a
cleaner manifest where each audio source appears once with all available
captions. It can also embed captions with CLAP and write source-level
embeddings for curriculum pair sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

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


def normalize_caption(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def compare_key(text: str) -> str:
    text = normalize_caption(text).lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dedupe_keep_order(items: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for item in items:
        caption = normalize_caption(item)
        if not caption:
            continue
        key = compare_key(caption)
        if key in seen:
            continue
        seen.add(key)
        kept.append(caption)
    return kept


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def summarize_counts(counts: list[int]) -> dict[str, float | int]:
    if not counts:
        return {"min": 0, "mean": 0, "median": 0, "max": 0}
    ordered = sorted(counts)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median: float | int = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "min": min(counts),
        "mean": round(sum(counts) / len(counts), 3),
        "median": median,
        "max": max(counts),
    }


def resolve_audio_path(audio_roots: list[str], wav: str) -> str:
    if not audio_roots:
        return wav

    basename = os.path.basename(wav)
    for root in audio_roots:
        candidate = Path(root).expanduser() / basename
        if candidate.exists():
            return str(candidate)
    return str(Path(audio_roots[0]).expanduser() / basename)


def is_fsd_dataset(dataset: str) -> bool:
    return dataset.startswith("fsd50k")


def source_id(dataset: str, wav: str, ordinal: int) -> str:
    return f"{dataset}:{Path(wav).name}:{ordinal}"


def load_fsd_sources(path: Path, dataset: str, audio_roots: list[str]) -> list[dict[str, Any]]:
    rows = read_json(path)["data"]
    sources: list[dict[str, Any]] = []

    for ordinal, row in enumerate(rows):
        captions = dedupe_keep_order(
            [row.get("caption", ""), *row.get("paraphrases", []), *row.get("captions", [])]
        )
        if not captions:
            continue

        wav = resolve_audio_path(audio_roots, str(row["wav"]))
        labels = [str(label) for label in row.get("labels", []) if label]
        specific_labels = [
            str(label)
            for label in row.get("specific_labels", [])
            if label and str(label) not in BROAD_FSD_LABELS
        ]
        if not specific_labels:
            specific_labels = [label for label in labels if label not in BROAD_FSD_LABELS]

        sources.append(
            {
                "source_id": source_id(dataset, wav, ordinal),
                "dataset": dataset,
                "split": row.get("split", ""),
                "wav": wav,
                "wav_name": Path(wav).name,
                "captions": captions,
                "caption_count": len(captions),
                "labels": labels,
                "specific_labels": specific_labels,
            }
        )

    return sources


def load_fsd_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    by_wav: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            labels = [label for label in row.get("labels", "").split(",") if label]
            by_wav[f"{row['fname']}.wav"] = {
                "split": row.get("split", ""),
                "labels": labels,
                "specific_labels": [label for label in labels if label not in BROAD_FSD_LABELS],
            }
    return by_wav


def apply_fsd_ground_truth(sources: list[dict[str, Any]], gt_csv: Path) -> None:
    gt = load_fsd_ground_truth(gt_csv)
    if not gt:
        return

    for source in sources:
        meta = gt.get(source["wav_name"])
        if not meta:
            continue
        if not source.get("labels"):
            source["labels"] = meta["labels"]
        if not source.get("specific_labels"):
            source["specific_labels"] = meta["specific_labels"]
        if not source.get("split"):
            source["split"] = meta.get("split", "")


def load_clotho_sources(path: Path, audio_roots: list[str]) -> list[dict[str, Any]]:
    rows = read_json(path)["data"]
    grouped: dict[str, list[object]] = defaultdict(list)

    for row in rows:
        wav = resolve_audio_path(audio_roots, str(row["wav"]))
        captions: list[object] = []
        if row.get("caption"):
            captions.append(row["caption"])
        captions.extend(row.get("paraphrases", []))
        captions.extend(row.get("captions", []))
        grouped[wav].extend(captions)

    sources: list[dict[str, Any]] = []
    for ordinal, (wav, captions) in enumerate(grouped.items()):
        unique_captions = dedupe_keep_order(captions)
        if not unique_captions:
            continue
        sources.append(
            {
                "source_id": source_id("clotho", wav, ordinal),
                "dataset": "clotho",
                "split": "",
                "wav": wav,
                "wav_name": Path(wav).name,
                "captions": unique_captions,
                "caption_count": len(unique_captions),
                "labels": [],
                "specific_labels": [],
            }
        )

    return sources


def filter_existing_audio(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    missing_by_parent: Counter[str] = Counter()
    missing_examples: list[str] = []

    for source in sources:
        wav = source["wav"]
        if os.path.exists(wav):
            kept.append(source)
            continue
        missing_by_parent[str(Path(wav).parent)] += 1
        if len(missing_examples) < 20:
            missing_examples.append(wav)

    return kept, {
        "input_sources": len(sources),
        "kept_sources": len(kept),
        "missing_sources": len(sources) - len(kept),
        "missing_by_parent": dict(missing_by_parent.most_common(20)),
        "missing_examples": missing_examples,
    }


def caption_rows_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        source["embedding_index"] = source_index
        source["caption_embedding_start"] = len(rows)
        for caption_index, caption in enumerate(source["captions"]):
            rows.append(
                {
                    "source_index": source_index,
                    "source_id": source["source_id"],
                    "caption_index": caption_index,
                    "caption": caption,
                }
            )
        source["caption_embedding_count"] = len(source["captions"])
    return rows


def tensor_from_clap_output(output: Any) -> Any:
    if hasattr(output, "norm"):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    raise TypeError(f"Unsupported CLAP text output type: {type(output)}")


def choose_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_captions(
    caption_rows: list[dict[str, Any]],
    source_count: int,
    out_dir: Path,
    model_name: str,
    batch_size: int,
    device: str,
    local_files_only: bool,
    write_caption_embeddings: bool,
) -> dict[str, Any]:
    import torch
    from transformers import ClapModel, ClapProcessor

    device = choose_device(device)
    processor = ClapProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = ClapModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval().to(device)

    source_sums = np.zeros((source_count, 512), dtype=np.float32)
    source_counts = np.zeros(source_count, dtype=np.int32)

    caption_embedding_path = out_dir / "caption_embeddings.npy"
    caption_embeddings = None
    if write_caption_embeddings:
        caption_embeddings = np.lib.format.open_memmap(
            caption_embedding_path,
            mode="w+",
            dtype="float32",
            shape=(len(caption_rows), 512),
        )

    with torch.no_grad():
        for start in range(0, len(caption_rows), batch_size):
            batch_rows = caption_rows[start : start + batch_size]
            texts = [row["caption"] for row in batch_rows]
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            embeddings = tensor_from_clap_output(model.get_text_features(**inputs))
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            embeddings_np = embeddings.cpu().numpy().astype("float32")

            if caption_embeddings is not None:
                caption_embeddings[start : start + len(batch_rows)] = embeddings_np

            for offset, row in enumerate(batch_rows):
                source_index = int(row["source_index"])
                source_sums[source_index] += embeddings_np[offset]
                source_counts[source_index] += 1

            done = min(start + batch_size, len(caption_rows))
            if start == 0 or done == len(caption_rows) or done % (batch_size * 50) == 0:
                print(f"embedded captions: {done:,}/{len(caption_rows):,}", flush=True)

    source_embeddings = source_sums / np.maximum(source_counts[:, None], 1)
    norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    source_embeddings = source_embeddings / np.maximum(norms, 1e-12)
    source_embedding_path = out_dir / "source_embeddings.npy"
    np.save(source_embedding_path, source_embeddings.astype("float32"))

    if caption_embeddings is not None:
        caption_embeddings.flush()

    return {
        "model": model_name,
        "device": device,
        "source_embeddings": str(source_embedding_path),
        "caption_embeddings": str(caption_embedding_path) if write_caption_embeddings else None,
        "caption_embedding_count": len(caption_rows) if write_caption_embeddings else 0,
        "source_embedding_mode": "normalized_mean_of_caption_embeddings",
    }


def label_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not (is_fsd_dataset(left["dataset"]) and is_fsd_dataset(right["dataset"])):
        return False
    left_labels = set(left.get("specific_labels") or left.get("labels") or [])
    right_labels = set(right.get("specific_labels") or right.get("labels") or [])
    return bool(left_labels & right_labels)


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def summarize_similarities(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p01": percentile(values, 1),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def sample_similarity_summary(
    sources: list[dict[str, Any]],
    embeddings: np.ndarray,
    pair_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    sims: list[float] = []
    safe_sims: list[float] = []
    same_label_sims: list[float] = []

    for _ in range(pair_count):
        i = rng.randrange(len(sources))
        j = rng.randrange(len(sources) - 1)
        if j >= i:
            j += 1
        sim = float(np.dot(embeddings[i], embeddings[j]))
        sims.append(sim)
        if label_overlap(sources[i], sources[j]):
            same_label_sims.append(sim)
        else:
            safe_sims.append(sim)

    summary = {
        "pair_count": pair_count,
        "all_pairs": summarize_similarities(sims),
        "safe_pairs_no_fsd_specific_overlap": summarize_similarities(safe_sims),
    }
    if same_label_sims:
        summary["fsd_same_specific_label_pairs"] = summarize_similarities(same_label_sims)
    summary["suggested_thresholds"] = {
        "t1_transformer_warmup_easy": percentile(safe_sims, 25),
        "t2_full_easy": percentile(safe_sims, 40),
        "t3_full_medium": percentile(safe_sims, 60),
        "t4_full_hard": percentile(safe_sims, 75),
        "t5_full_hard_upper": percentile(safe_sims, 90),
    }
    return summary


def build_sources(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    input_report: dict[str, Any] = {}

    fsd_dev_sources = load_fsd_sources(
        args.fsd_dev_json,
        dataset="fsd50k_dev",
        audio_roots=[root for root in [args.fsd_dev_audio_root, args.fsd_audio_root] if root],
    )
    apply_fsd_ground_truth(fsd_dev_sources, args.fsd_dev_gt_csv)
    sources.extend(fsd_dev_sources)
    input_report["fsd_dev"] = {"path": str(args.fsd_dev_json), "sources": len(fsd_dev_sources)}

    if args.include_fsd_eval:
        fsd_eval_sources = load_fsd_sources(
            args.fsd_eval_json,
            dataset="fsd50k_eval",
            audio_roots=[root for root in [args.fsd_eval_audio_root, args.fsd_audio_root] if root],
        )
        apply_fsd_ground_truth(fsd_eval_sources, args.fsd_eval_gt_csv)
        sources.extend(fsd_eval_sources)
        input_report["fsd_eval"] = {"path": str(args.fsd_eval_json), "sources": len(fsd_eval_sources)}

    clotho_roots = [
        *args.clotho_all_audio_root,
        args.clotho_dev_audio_root,
        args.clotho_val_audio_root,
        args.clotho_eval_audio_root,
        args.clotho_audio_root,
    ]
    clotho_roots = [root for root in clotho_roots if root]
    clotho_sources = load_clotho_sources(args.clotho_json, audio_roots=clotho_roots)
    sources.extend(clotho_sources)
    input_report["clotho"] = {"path": str(args.clotho_json), "sources": len(clotho_sources)}

    if args.limit_sources:
        sources = sources[: args.limit_sources]
        input_report["limit_sources"] = args.limit_sources

    return sources, input_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-dev-json", type=Path, default=SCRIPT_DIR / "../../upload/fsd50k_dev_train_val_auto_caption_gemini_paraphrases.json")
    parser.add_argument("--fsd-eval-json", type=Path, default=SCRIPT_DIR / "../../upload/fsd50k_eval_auto_caption_gemini_paraphrases.json")
    parser.add_argument("--clotho-json", type=Path, default=SCRIPT_DIR / "../../upload/clotho_all_public_16k_group_aug_merged.json")
    parser.add_argument("--fsd-dev-gt-csv", type=Path, default=SCRIPT_DIR / "../../paraphrase/FSD50K.ground_truth/dev.csv")
    parser.add_argument("--fsd-eval-gt-csv", type=Path, default=SCRIPT_DIR / "../../paraphrase/FSD50K.ground_truth/eval.csv")
    parser.add_argument("--fsd-audio-root", default=None)
    parser.add_argument("--fsd-dev-audio-root", default=None)
    parser.add_argument("--fsd-eval-audio-root", default=None)
    parser.add_argument("--clotho-audio-root", default=None)
    parser.add_argument("--clotho-all-audio-root", action="append", default=[])
    parser.add_argument("--clotho-dev-audio-root", default=None)
    parser.add_argument("--clotho-val-audio-root", default=None)
    parser.add_argument("--clotho-eval-audio-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "datafiles/curriculum")
    parser.add_argument("--include-fsd-eval", action="store_true", default=True)
    parser.add_argument("--no-include-fsd-eval", dest="include_fsd_eval", action="store_false")
    parser.add_argument("--drop-missing-audio", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--model", default="laion/clap-htsat-unfused")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--write-caption-embeddings", action="store_true")
    parser.add_argument("--pair-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit-sources", type=int, default=0)
    args = parser.parse_args()

    sources, input_report = build_sources(args)
    missing_report = None
    if args.drop_missing_audio:
        sources, missing_report = filter_existing_audio(sources)

    caption_rows = caption_rows_from_sources(sources)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = args.out_dir / "sources.json"
    caption_manifest_path = args.out_dir / "captions.json"
    stats_path = args.out_dir / "curriculum_stats.json"

    write_json(source_manifest_path, {"data": sources})
    write_json(caption_manifest_path, {"data": caption_rows})

    dataset_counts = Counter(source["dataset"] for source in sources)
    caption_counts = [len(source["captions"]) for source in sources]
    stats: dict[str, Any] = {
        "seed": args.seed,
        "inputs": input_report,
        "outputs": {
            "sources": str(source_manifest_path),
            "captions": str(caption_manifest_path),
            "source_embeddings": None,
            "caption_embeddings": None,
        },
        "sources": len(sources),
        "captions": len(caption_rows),
        "sources_by_dataset": dict(dataset_counts),
        "captions_per_source": summarize_counts(caption_counts),
    }
    if missing_report:
        stats["missing_audio_filter"] = missing_report

    if not args.no_embed:
        embedding_report = embed_captions(
            caption_rows=caption_rows,
            source_count=len(sources),
            out_dir=args.out_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=args.local_files_only,
            write_caption_embeddings=args.write_caption_embeddings,
        )
        stats["embedding"] = embedding_report
        stats["outputs"]["source_embeddings"] = embedding_report["source_embeddings"]
        stats["outputs"]["caption_embeddings"] = embedding_report["caption_embeddings"]

        if args.pair_samples > 0:
            embeddings = np.load(embedding_report["source_embeddings"])
            stats["similarity"] = sample_similarity_summary(
                sources=sources,
                embeddings=embeddings,
                pair_count=args.pair_samples,
                seed=args.seed,
            )

    write_json(stats_path, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
