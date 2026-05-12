import json
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import IterableDataset, get_worker_info


def _is_fsd(source: dict[str, Any]) -> bool:
    return str(source.get("dataset", "")).startswith("fsd50k")


def _energy(waveform: torch.Tensor) -> torch.Tensor:
    return torch.mean(waveform ** 2)


def _rescale_to_match_energy(audio: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    ratio = (_energy(audio) / torch.clamp(_energy(reference), min=1e-10)) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return audio / ratio


def _dynamic_loudnorm(
    audio: torch.Tensor,
    reference: torch.Tensor,
    rng: random.Random,
    lower_db: int,
    higher_db: int,
) -> torch.Tensor:
    audio = _rescale_to_match_energy(audio, reference)
    gain = np.power(10.0, rng.randint(lower_db, higher_db) / 20.0)
    return float(gain) * audio


class CurriculumAudioTextDataset(IterableDataset):
    """Yield AudioSep examples with curriculum-aware source pairing.

    Each item already contains a mixture and target segment, so the training
    step should not use the old batch-neighbor SegmentMixer for this dataset.
    """

    def __init__(
        self,
        manifest_path: str,
        source_embeddings_path: str,
        sampling_rate: int = 16000,
        max_clip_len: int = 10,
        lower_db: int = -10,
        higher_db: int = 10,
        curriculum: dict[str, Any] | None = None,
        batch_size_per_device: int = 32,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.source_embeddings_path = source_embeddings_path
        self.sampling_rate = sampling_rate
        self.max_length = int(max_clip_len * sampling_rate)
        self.lower_db = int(lower_db)
        self.higher_db = int(higher_db)
        self.curriculum = curriculum or {}
        self.batch_size_per_device = max(int(batch_size_per_device), 1)

        with open(manifest_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self.sources = payload["data"] if isinstance(payload, dict) else payload
        if not self.sources:
            raise ValueError(f"No sources found in curriculum manifest: {manifest_path}")

        self.embeddings = np.load(source_embeddings_path).astype("float32")
        if self.embeddings.shape[0] != len(self.sources):
            raise ValueError(
                f"Embedding/source count mismatch: {self.embeddings.shape[0]} embeddings "
                f"for {len(self.sources)} sources"
            )
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

        self.label_sets = [
            set(source.get("specific_labels") or source.get("labels") or [])
            for source in self.sources
        ]
        self.base_seed = int(self.curriculum.get("seed", self.curriculum.get("base_seed", 1234)))
        self.initial_global_step = int(
            self.curriculum.get("initial_global_step", self.curriculum.get("start_step", 0))
        )
        self.max_sampling_attempts = int(self.curriculum.get("max_sampling_attempts", 300))
        self.audio_load_attempts = int(self.curriculum.get("audio_load_attempts", 20))
        self.default_negative_ratio = float(self.curriculum.get("negative_ratio", 0.0))
        self.default_negative_max_similarity = float(
            self.curriculum.get("negative_max_similarity", 0.05)
        )
        self.default_min_similarity = self.curriculum.get("min_similarity", None)
        self.default_max_similarity = float(self.curriculum.get("max_similarity", 0.01))
        total_steps = self.curriculum.get("total_steps", None)
        self.total_steps = int(total_steps) if total_steps is not None else None
        self.default_pair_bands = self._parse_pair_bands(
            self.curriculum.get("pair_bands", None),
            fallback_min_similarity=self.default_min_similarity,
            fallback_max_similarity=self.default_max_similarity,
        )
        self.stages = self._parse_stages(self.curriculum.get("stages", []))

    def _parse_pair_bands(
        self,
        pair_bands: list[dict[str, Any]] | None,
        fallback_min_similarity: float | None,
        fallback_max_similarity: float,
    ) -> list[dict[str, Any]]:
        if not pair_bands:
            return [
                {
                    "name": "default",
                    "weight": 1.0,
                    "min_similarity": fallback_min_similarity,
                    "max_similarity": float(fallback_max_similarity),
                }
            ]

        parsed = []
        for idx, band in enumerate(pair_bands):
            weight = float(band.get("weight", 1.0))
            if weight <= 0:
                continue
            parsed.append(
                {
                    "name": band.get("name", f"band_{idx + 1}"),
                    "weight": weight,
                    "min_similarity": band.get(
                        "min_similarity",
                        fallback_min_similarity,
                    ),
                    "max_similarity": float(
                        band.get("max_similarity", fallback_max_similarity)
                    ),
                }
            )

        if not parsed:
            raise ValueError("At least one curriculum pair band must have positive weight.")
        return parsed

    def _parse_stages(self, stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed = []
        cumulative_steps = 0
        cumulative_portion = 0.0
        for idx, stage in enumerate(stages):
            steps = stage.get("steps", None)
            portion = stage.get("portion", stage.get("ratio", stage.get("fraction", None)))
            if steps is not None:
                cumulative_steps += int(steps)
                end_step = cumulative_steps
            elif portion is not None:
                if self.total_steps is None or self.total_steps <= 0:
                    raise ValueError(
                        "Curriculum stage portions require total_steps. "
                        "Pass --num_steps to train.py."
                    )
                cumulative_portion += float(portion)
                end_step = int(round(cumulative_portion * self.total_steps))
                cumulative_steps = end_step
            else:
                end_step = None
            stage_min_similarity = stage.get("min_similarity", self.default_min_similarity)
            stage_max_similarity = float(stage.get("max_similarity", self.default_max_similarity))
            parsed.append(
                {
                    "name": stage.get("name", f"stage_{idx + 1}"),
                    "end_step": end_step,
                    "min_similarity": stage_min_similarity,
                    "max_similarity": stage_max_similarity,
                    "pair_bands": self._parse_pair_bands(
                        stage.get("pair_bands", None),
                        fallback_min_similarity=stage_min_similarity,
                        fallback_max_similarity=stage_max_similarity,
                    ),
                    "negative_ratio": float(
                        stage.get("negative_ratio", self.default_negative_ratio)
                    ),
                    "negative_max_similarity": float(
                        stage.get(
                            "negative_max_similarity",
                            self.default_negative_max_similarity,
                        )
                    ),
                }
            )
        return parsed

    def __len__(self) -> int:
        epoch_steps = int(self.curriculum.get("epoch_steps", 10_000))
        return epoch_steps * self.batch_size_per_device

    def _stage_for_sample(self, sample_index: int) -> dict[str, Any]:
        if not self.stages:
            return {
                "name": "static",
                "min_similarity": self.default_min_similarity,
                "max_similarity": self.default_max_similarity,
                "pair_bands": self.default_pair_bands,
                "negative_ratio": self.default_negative_ratio,
                "negative_max_similarity": self.default_negative_max_similarity,
            }

        local_step = sample_index // self.batch_size_per_device
        finite_ends = [
            stage["end_step"] if stage["end_step"] is not None else float("inf")
            for stage in self.stages
        ]
        idx = min(bisect_right(finite_ends, local_step), len(self.stages) - 1)
        return self.stages[idx]

    def _choose_pair_band(
        self,
        rng: random.Random,
        stage: dict[str, Any],
    ) -> dict[str, Any]:
        pair_bands = stage.get("pair_bands") or self.default_pair_bands
        total_weight = sum(float(band["weight"]) for band in pair_bands)
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for band in pair_bands:
            cumulative += float(band["weight"])
            if threshold <= cumulative:
                return band
        return pair_bands[-1]

    def _same_or_bad_pair(self, left_idx: int, right_idx: int, max_similarity: float) -> bool:
        if left_idx == right_idx:
            return True

        left = self.sources[left_idx]
        right = self.sources[right_idx]
        if left.get("source_id") == right.get("source_id"):
            return True
        if left.get("wav") == right.get("wav"):
            return True

        if float(np.dot(self.embeddings[left_idx], self.embeddings[right_idx])) > max_similarity:
            return True

        if _is_fsd(left) and _is_fsd(right) and self.label_sets[left_idx] & self.label_sets[right_idx]:
            return True

        return False

    def _similarity_in_range(
        self,
        left_idx: int,
        right_idx: int,
        min_similarity: float | None,
        max_similarity: float,
    ) -> bool:
        sim = float(np.dot(self.embeddings[left_idx], self.embeddings[right_idx]))
        if sim > max_similarity:
            return False
        if min_similarity is not None and sim < float(min_similarity):
            return False
        return True

    def _sample_positive_pair(
        self,
        rng: random.Random,
        min_similarity: float | None,
        max_similarity: float,
    ) -> tuple[int, int]:
        n = len(self.sources)

        for _ in range(self.max_sampling_attempts):
            left_idx = rng.randrange(n)
            right_idx = rng.randrange(n - 1)
            if right_idx >= left_idx:
                right_idx += 1
            if self._same_or_bad_pair(left_idx, right_idx, max_similarity):
                continue
            if self._similarity_in_range(left_idx, right_idx, min_similarity, max_similarity):
                return left_idx, right_idx

        raise RuntimeError(
            "Could not sample a valid source pair. Consider raising max_similarity "
            "or max_sampling_attempts, or relaxing the selected pair band."
        )

    def _sample_negative_query(
        self,
        rng: random.Random,
        left_idx: int,
        right_idx: int,
        max_similarity: float,
    ) -> int:
        n = len(self.sources)
        left = self.sources[left_idx]
        right = self.sources[right_idx]

        for _ in range(self.max_sampling_attempts):
            query_idx = rng.randrange(n)
            if query_idx in {left_idx, right_idx}:
                continue
            query = self.sources[query_idx]
            if query.get("wav") in {left.get("wav"), right.get("wav")}:
                continue
            if float(np.dot(self.embeddings[query_idx], self.embeddings[left_idx])) > max_similarity:
                continue
            if float(np.dot(self.embeddings[query_idx], self.embeddings[right_idx])) > max_similarity:
                continue
            if _is_fsd(query) and _is_fsd(left) and self.label_sets[query_idx] & self.label_sets[left_idx]:
                continue
            if _is_fsd(query) and _is_fsd(right) and self.label_sets[query_idx] & self.label_sets[right_idx]:
                continue
            return query_idx

        raise RuntimeError(
            "Could not sample a negative query. Consider raising negative_max_similarity "
            "or max_sampling_attempts."
        )

    def _cut_or_randomcrop(self, waveform: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if waveform.size(1) > self.max_length:
            random_idx = rng.randint(0, waveform.size(1) - self.max_length)
            waveform = waveform[:, random_idx : random_idx + self.max_length]
        else:
            padded = torch.zeros(1, self.max_length)
            padded[:, : waveform.size(1)] = waveform
            waveform = padded

        return waveform

    def _read_audio(self, source_idx: int, rng: random.Random) -> torch.Tensor:
        source = self.sources[source_idx]
        audio_path = source["wav"]
        audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)

        if audio_data.size(1) < self.sampling_rate:
            raise RuntimeError(f"{audio_path} is too short")

        if audio_data.shape[0] > 1:
            audio_data = torch.mean(audio_data, dim=0)
        else:
            audio_data = audio_data.squeeze(0)

        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(
                audio_data,
                orig_freq=audio_rate,
                new_freq=self.sampling_rate,
            )

        return self._cut_or_randomcrop(audio_data.unsqueeze(0), rng)

    def _choose_caption(self, source_idx: int, rng: random.Random) -> str:
        captions = self.sources[source_idx].get("captions", [])
        if not captions:
            raise RuntimeError(f"Source has no captions: {self.sources[source_idx]}")
        return rng.choice(captions)

    def _make_example(self, rng: random.Random, stage: dict[str, Any]) -> dict[str, Any]:
        pair_band = self._choose_pair_band(rng, stage)
        min_similarity = pair_band.get("min_similarity", None)
        max_similarity = float(pair_band["max_similarity"])
        negative_ratio = float(stage.get("negative_ratio", 0.0))
        negative_max_similarity = float(
            stage.get("negative_max_similarity", self.default_negative_max_similarity)
        )

        last_error: Exception | None = None
        for _ in range(self.audio_load_attempts):
            try:
                target_idx, noise_idx = self._sample_positive_pair(
                    rng=rng,
                    min_similarity=min_similarity,
                    max_similarity=max_similarity,
                )
                pair_similarity = float(
                    np.dot(self.embeddings[target_idx], self.embeddings[noise_idx])
                )
                is_negative = rng.random() < negative_ratio
                query_idx = target_idx
                if is_negative:
                    query_idx = self._sample_negative_query(
                        rng=rng,
                        left_idx=target_idx,
                        right_idx=noise_idx,
                        max_similarity=negative_max_similarity,
                    )

                target_audio = self._read_audio(target_idx, rng)
                noise_audio = self._read_audio(noise_idx, rng)
                noise_audio = _dynamic_loudnorm(
                    audio=noise_audio,
                    reference=target_audio,
                    rng=rng,
                    lower_db=self.lower_db,
                    higher_db=self.higher_db,
                )
                mixture = target_audio + noise_audio
                segment = torch.zeros_like(target_audio) if is_negative else target_audio.clone()

                max_value = torch.max(torch.abs(mixture))
                if max_value > 1:
                    mixture = mixture * (0.9 / max_value)
                    segment = segment * (0.9 / max_value)

                return {
                    "text": self._choose_caption(query_idx, rng),
                    "waveform": segment,
                    "segment": segment,
                    "mixture": mixture,
                    "is_negative": float(is_negative),
                    "stage": stage["name"],
                    "pair_band": pair_band["name"],
                    "pair_similarity": pair_similarity,
                    "is_easy_pair": float(pair_band["name"] == "easy"),
                    "is_medium_pair": float(pair_band["name"] == "medium"),
                    "is_hard_pair": float(pair_band["name"] == "hard"),
                    "target_source_id": self.sources[target_idx]["source_id"],
                    "noise_source_id": self.sources[noise_idx]["source_id"],
                    "query_source_id": self.sources[query_idx]["source_id"],
                    "modality": "audio_text",
                }
            except Exception as exc:  # noqa: BLE001 - retry a new sampled example.
                last_error = exc

        raise RuntimeError(f"Failed to build curriculum example: {last_error}") from last_error

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        rng = random.Random(
            self.base_seed
            + self.initial_global_step * 97
            + rank * 1_000_003
            + worker_id * 10_007
        )
        sample_index = self.initial_global_step * self.batch_size_per_device + worker_id

        while True:
            stage = self._stage_for_sample(sample_index)
            yield self._make_example(rng, stage)
            sample_index += num_workers
