"""Domain classification and weight construction for YOLOX training data."""
from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Mapping, Sequence

import torch


DOMAIN_NAMES = ("coco", "day", "night", "twilight", "mori")


def classify_domain(file_name: str) -> str:
    """Classify the configured train_all filename convention.

    Names without one of the custom-domain prefixes are currently treated as
    COCO.  This preserves compatibility with zero-padded COCO image names.
    """
    if file_name.startswith("day_"):
        return "day"
    if file_name.startswith("night_"):
        return "night"
    if file_name.startswith("twilight_"):
        return "twilight"
    if file_name.startswith("mori_"):
        return "mori"
    return "coco"


def build_image_weights(
    file_names: Sequence[str], probabilities: Mapping[str, float]
) -> tuple[torch.Tensor, dict[str, int], list[str]]:
    """Return dataset-index-aligned image weights and domain metadata."""
    domains = [classify_domain(file_name) for file_name in file_names]
    domain_counter = Counter(domains)
    counts = {name: domain_counter[name] for name in DOMAIN_NAMES}
    _validate_probabilities(probabilities, counts)

    weights = torch.tensor(
        [float(probabilities[domain]) / counts[domain] for domain in domains],
        dtype=torch.double,
    )
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("computed image weights must be finite and non-negative")
    if weights.sum().item() <= 0:
        raise ValueError("the sum of computed image weights must be greater than zero")
    return weights, counts, domains


def build_dataset_image_weights(dataset, probabilities: Mapping[str, float]):
    """Build weights in exactly the order consumed by a wrapped COCODataset."""
    base_dataset = dataset
    # YOLOX MosaicDetection stores the wrapped dataset as ``_dataset``;
    # torchvision-style wrappers commonly use ``dataset``.
    while hasattr(base_dataset, "dataset") or hasattr(base_dataset, "_dataset"):
        base_dataset = getattr(base_dataset, "dataset", getattr(base_dataset, "_dataset", None))
    if not hasattr(base_dataset, "ids") or not hasattr(base_dataset, "coco"):
        raise TypeError("expected a COCODataset or a wrapper around COCODataset")

    image_infos = base_dataset.coco.loadImgs(base_dataset.ids)
    if len(image_infos) != len(base_dataset.ids):
        raise ValueError("could not load image metadata for every COCODataset index")
    return build_image_weights([image["file_name"] for image in image_infos], probabilities)


def _validate_probabilities(probabilities: Mapping[str, float], counts: Mapping[str, int]) -> None:
    keys = set(probabilities)
    unknown = keys - set(DOMAIN_NAMES)
    if unknown:
        raise ValueError(f"unknown domain probabilities: {sorted(unknown)}")
    missing = {domain for domain, count in counts.items() if count > 0} - keys
    if missing:
        raise ValueError(f"missing sampling probabilities for dataset domains: {sorted(missing)}")

    total = 0.0
    for domain, probability in probabilities.items():
        try:
            probability = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"probability for {domain!r} must be numeric") from exc
        if not isfinite(probability) or probability < 0:
            raise ValueError(f"probability for {domain!r} must be finite and non-negative")
        if probability > 0 and counts[domain] == 0:
            raise ValueError(
                f"{domain!r} has sampling probability {probability} but no images"
            )
        total += probability
    if abs(total - 1.0) >= 1e-6:
        raise ValueError(f"domain sampling probabilities must sum to 1.0; got {total}")
