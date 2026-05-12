#!/usr/bin/env python3
"""Paraphrase FSD50K auto captions with Gemini while preserving labels.

This script is different from label-to-caption generation. It uses the provided
auto caption as the semantic anchor, and uses FSD50K labels only as guardrails.

Output shape:
{
  "data": [
    {
      "wav": "10000.wav",
      "caption": "The act of breathing creates audible respiratory sounds.",
      "paraphrases": [
        "A person is breathing audibly.",
        "Audible breathing sounds are present."
      ],
      "labels": ["Breathing", "Respiratory_sounds", "Human_voice"]
    }
  ]
}

You can flatten this later into AudioSep rows by using the original caption plus
one or more paraphrases as alternative captions for the same wav.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "Task_provide_dataset"
FSD_GT_DIR = ROOT / "paraphrase" / "FSD50K.ground_truth"

DEFAULT_MODEL = "gemini-2.5-flash"

# Approximate standard paid-tier Gemini API pricing, USD / 1M tokens.
# Verify at https://ai.google.dev/gemini-api/docs/pricing before large runs.
PRICE_PER_MILLION = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}

BROAD_LABELS = {
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

BAD_PHRASES = [
    "maybe",
    "possibly",
    "probably",
    "appears",
    "seems",
    "might",
    "could be",
    "clip",
    "recording",
    "audio file",
    "fsd50k",
    "label",
]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    temp.replace(path)


def normalize_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().strip("\"'")
    if text and text[-1] not in ".!?":
        text += "."
    return text


def normalize_label(label: str) -> str:
    label = label.replace("_", " ")
    label = label.replace("(", " ").replace(")", " ")
    return re.sub(r"\s+", " ", label).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def is_good_caption(text: object, min_words: int, max_words: int) -> bool:
    if not isinstance(text, str):
        return False
    text = normalize_caption(text)
    if not text or text.isdigit():
        return False
    wc = word_count(text)
    if wc < min_words or wc > max_words:
        return False
    lower = text.lower()
    return not any(phrase in lower for phrase in BAD_PHRASES)


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    by_wav = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = [label for label in row["labels"].split(",") if label]
            by_wav[f"{row['fname']}.wav"] = {
                "labels": labels,
                "specific_labels": [label for label in labels if label not in BROAD_LABELS],
                "split": row.get("split", ""),
            }
    return by_wav


def load_items(auto_caption_json: Path, gt_csv: Path, split: str) -> list[dict[str, Any]]:
    gt = load_ground_truth(gt_csv)
    items = []
    for row in read_json(auto_caption_json)["data"]:
        wav = Path(row["wav"]).name
        caption = normalize_caption(str(row["caption"]))
        meta = gt.get(wav, {"labels": [], "specific_labels": [], "split": ""})
        if split in {"train", "val"} and meta.get("split") != split:
            continue
        if split == "eval" and meta.get("split", "") not in {"", "eval"}:
            continue
        if not is_good_caption(caption, min_words=2, max_words=30):
            continue
        items.append(
            {
                "wav": wav,
                "caption": caption,
                "labels": meta["labels"],
                "specific_labels": meta["specific_labels"],
                "split": meta.get("split", ""),
            }
        )
    return items


def compact_labels(labels: list[str]) -> list[str]:
    specific = [label for label in labels if label not in BROAD_LABELS]
    chosen = specific if specific else labels
    return [normalize_label(label) for label in chosen[:8]]


def build_prompt(items: list[dict[str, Any]], paraphrases_per_item: int) -> str:
    payload = []
    for item in items:
        payload.append(
            {
                "wav": item["wav"],
                "original_caption": item["caption"],
                "target_labels": compact_labels(item["specific_labels"]),
                "all_labels": [normalize_label(label) for label in item["labels"][:12]],
            }
        )

    return (
        "Rewrite audio captions for language-queried source separation training.\n"
        "The original caption is the semantic anchor. The target labels are guardrails.\n\n"
        f"For each item, generate exactly {paraphrases_per_item} paraphrases.\n\n"
        "Rules:\n"
        "- Preserve the target sound identity exactly.\n"
        "- Do not introduce new sound sources, locations, speakers, emotions, or visual details.\n"
        "- Do not remove specific target information from the original caption.\n"
        "- If the original caption is broad, stay broad; do not guess a more specific sound.\n"
        "- Use natural LASS-style audio-query language.\n"
        "- Each paraphrase should be one sentence, 5 to 14 words when possible.\n"
        "- Make paraphrases lexically diverse but semantically conservative.\n"
        "- Avoid generic text like 'a sound is heard' unless the original is also generic.\n"
        "- Do not mention labels, classes, FSD50K, clips, files, or recordings.\n"
        "- Return valid JSON only: an array of objects with keys wav and paraphrases.\n\n"
        "Good examples:\n"
        "[{\"wav\":\"1.wav\",\"original_caption\":\"A dog barks.\","
        "\"target_labels\":[\"Bark\",\"Dog\"],\"all_labels\":[\"Bark\",\"Dog\",\"Animal\"]}]\n"
        "=> [{\"wav\":\"1.wav\",\"paraphrases\":["
        "\"A dog is barking loudly.\","
        "\"A barking dog can be heard.\","
        "\"The dog keeps barking.\""
        "]}]\n\n"
        "[{\"wav\":\"2.wav\",\"original_caption\":\"A bowed string instrument produces music.\","
        "\"target_labels\":[\"Bowed string instrument\"],"
        "\"all_labels\":[\"Bowed string instrument\",\"Musical instrument\",\"Music\"]}]\n"
        "=> [{\"wav\":\"2.wav\",\"paraphrases\":["
        "\"A bowed string instrument is playing music.\","
        "\"Music comes from a bowed string instrument.\","
        "\"A bowed string instrument plays a melody.\""
        "]}]\n\n"
        "Bad examples:\n"
        "- Adding 'outside', 'street', or 'crowd' when absent from the original.\n"
        "- Changing 'vehicle' into 'race car' unless the original says race car.\n"
        "- Changing 'instrument' into 'guitar' unless labels/original support guitar.\n\n"
        "Items:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "maxOutputTokens": max_output_tokens,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {payload}") from exc

    parsed = json.loads(strip_json_fence(text))
    if not isinstance(parsed, list):
        raise RuntimeError(f"Gemini returned non-list JSON: {parsed!r}")
    return parsed


def clean_paraphrases(
    returned_rows: list[dict[str, Any]],
    batch_items: list[dict[str, Any]],
    paraphrases_per_item: int,
    min_words: int,
    max_words: int,
) -> dict[str, list[str]]:
    originals = {item["wav"]: item["caption"] for item in batch_items}
    expected = set(originals)
    output = {}

    for row in returned_rows:
        if not isinstance(row, dict):
            continue
        wav = row.get("wav")
        if wav not in expected:
            continue
        raw_phrases = row.get("paraphrases", [])
        if not isinstance(raw_phrases, list):
            continue
        phrases = []
        seen = {originals[wav].lower()}
        for phrase in raw_phrases:
            phrase = normalize_caption(str(phrase))
            key = phrase.lower()
            if key in seen:
                continue
            if not is_good_caption(phrase, min_words=min_words, max_words=max_words):
                continue
            seen.add(key)
            phrases.append(phrase)
        output[wav] = phrases[:paraphrases_per_item]

    missing = expected - set(output)
    if missing:
        raise RuntimeError(f"Missing paraphrases for wavs: {sorted(missing)[:5]}")
    return output


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = read_json(path).get("data", [])
    return {
        row["wav"]: row
        for row in data
        if isinstance(row, dict) and isinstance(row.get("wav"), str)
    }


def chunks(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def approximate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def estimate_cost(items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    input_tokens = 0
    output_tokens = 0
    for batch in chunks(items, args.batch_size):
        input_tokens += approximate_tokens(build_prompt(batch, args.paraphrases_per_item))
        output_tokens += len(batch) * args.paraphrases_per_item * 18

    price = PRICE_PER_MILLION.get(args.model, PRICE_PER_MILLION[DEFAULT_MODEL])
    input_cost = input_tokens / 1_000_000 * price["input"]
    output_cost = output_tokens / 1_000_000 * price["output"]
    print(f"items: {len(items):,}")
    print(f"paraphrases_per_item: {args.paraphrases_per_item}")
    print(f"estimated_input_tokens: {input_tokens:,}")
    print(f"estimated_output_tokens: {output_tokens:,}")
    print(f"model: {args.model}")
    print(f"estimated_standard_api_cost_usd: ${input_cost + output_cost:.2f}")
    print(f"  input: ${input_cost:.2f}")
    print(f"  output: ${output_cost:.2f}")


def output_payload(items: list[dict[str, Any]], generated: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in items:
        row = generated.get(item["wav"])
        if row is None:
            continue
        rows.append(row)
    return {"data": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-caption-json", type=Path, default=None)
    parser.add_argument("--gt-csv", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "dev", "eval", "all"],
        default="train",
        help="'train' and 'val' are subsets of FSD50K dev; 'eval' uses the separate FSD50K eval set.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--paraphrases-per-item", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel Gemini requests. Start with 4-8 to avoid rate limits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_split_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.split == "eval":
        if args.auto_caption_json is None:
            args.auto_caption_json = TASK_DIR / "fsd50k_eval_auto_caption.json"
        if args.gt_csv is None:
            args.gt_csv = FSD_GT_DIR / "eval.csv"
        if args.out is None:
            args.out = ROOT / "para" / "fsd50k_eval_auto_caption_gemini_paraphrases.json"
    else:
        if args.auto_caption_json is None:
            args.auto_caption_json = TASK_DIR / "fsd50k_dev_auto_caption.json"
        if args.gt_csv is None:
            args.gt_csv = FSD_GT_DIR / "dev.csv"
        if args.out is None:
            split_name = "dev" if args.split in {"dev", "all"} else args.split
            args.out = ROOT / "para" / f"fsd50k_{split_name}_auto_caption_gemini_paraphrases.json"
    return args


def main() -> None:
    args = apply_split_defaults(parse_args())
    items = load_items(args.auto_caption_json, args.gt_csv, args.split)
    if args.max_items is not None:
        items = items[: args.max_items]

    if args.estimate_only:
        estimate_cost(items, args)
        return

    if args.dry_run:
        print(build_prompt(items[: min(args.batch_size, len(items))], args.paraphrases_per_item))
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key. Set {args.api_key_env}.", file=sys.stderr)
        sys.exit(2)

    existing = {} if args.overwrite else load_existing(args.out)
    generated = dict(existing)
    pending = [item for item in items if item["wav"] not in generated]

    print(f"items: {len(items):,}")
    print(f"existing: {len(existing):,}")
    print(f"pending: {len(pending):,}")
    print(f"out: {args.out}")

    pending_batches = list(chunks(pending, args.batch_size))
    print(f"batches: {len(pending_batches):,}")
    print(f"workers: {args.workers}")

    if args.workers <= 1:
        bar = progress(pending_batches, desc="Paraphrasing batches", unit="batch")
        for batch_index, batch in enumerate(bar, start=1):
            paraphrases_by_wav = process_batch(batch_index, batch, args, api_key)
            merge_batch(items, generated, batch, paraphrases_by_wav)
            write_json(args.out, output_payload(items, generated))
            if tqdm is not None:
                bar.set_postfix(generated=f"{len(generated):,}/{len(items):,}", batch_size=len(batch))
            else:
                print(f"wrote {len(generated):,}/{len(items):,}")
            time.sleep(args.sleep)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_batch, batch_index, batch, args, api_key): (
                    batch_index,
                    batch,
                )
                for batch_index, batch in enumerate(pending_batches, start=1)
            }
            bar = progress(
                concurrent.futures.as_completed(futures),
                desc="Paraphrasing batches",
                unit="batch",
                total=len(futures),
            )
            for future in bar:
                batch_index, batch = futures[future]
                paraphrases_by_wav = future.result()
                merge_batch(items, generated, batch, paraphrases_by_wav)
                write_json(args.out, output_payload(items, generated))
                if tqdm is not None:
                    bar.set_postfix(generated=f"{len(generated):,}/{len(items):,}", last_batch=batch_index)
                else:
                    print(
                        f"completed batch {batch_index}; "
                        f"wrote {len(generated):,}/{len(items):,}"
                    )

    print(f"done: {args.out}")


def process_batch(
    batch_index: int,
    batch: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, list[str]]:
    prompt = build_prompt(batch, args.paraphrases_per_item)
    for attempt in range(args.retries):
        try:
            raw = call_gemini(
                prompt=prompt,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
            return clean_paraphrases(
                raw,
                batch,
                paraphrases_per_item=args.paraphrases_per_item,
                min_words=args.min_words,
                max_words=args.max_words,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            wait = min(60, 2**attempt)
            print(
                f"batch={batch_index} attempt={attempt + 1}/{args.retries} failed: {exc}; sleeping {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed batch {batch_index} after {args.retries} attempts")


def merge_batch(
    items: list[dict[str, Any]],
    generated: dict[str, dict[str, Any]],
    batch: list[dict[str, Any]],
    paraphrases_by_wav: dict[str, list[str]],
) -> None:
    del items
    for item in batch:
        generated[item["wav"]] = {
            "wav": item["wav"],
            "caption": item["caption"],
            "paraphrases": paraphrases_by_wav[item["wav"]],
            "labels": item["labels"],
            "specific_labels": item["specific_labels"],
            "split": item["split"],
        }


if __name__ == "__main__":
    main()
