"""Small, framework-free helpers for domain-weighted validation metrics."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


VALIDATION_DOMAINS = ("coco", "day", "night", "twilight", "mori")


def validate_domain_weights(
    weights: Mapping[str, float], domains: Sequence[str] = VALIDATION_DOMAINS
) -> dict[str, float]:
    """Validate a complete probability-like domain-weight configuration."""
    expected = set(domains)
    actual = set(weights)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"Unknown validation domain weight(s): {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing validation domain weight(s): {sorted(missing)}")
    result = {domain: float(weights[domain]) for domain in domains}
    invalid = [domain for domain, weight in result.items() if not math.isfinite(weight) or weight < 0]
    if invalid:
        raise ValueError(f"Validation weights must be finite and >= 0: {invalid}")
    total = sum(result.values())
    if abs(total - 1.0) >= 1e-6:
        raise ValueError(f"Validation domain weights must sum to 1.0, got {total:.12g}")
    return result


def aggregate_domain_values(
    domain_values: Mapping[str, float],
    weights: Mapping[str, float],
    image_counts: Mapping[str, int] | None = None,
    domains: Sequence[str] = VALIDATION_DOMAINS,
) -> float:
    """Weight already-averaged domain values; image counts are never multiplied in."""
    validated_weights = validate_domain_weights(weights, domains)
    missing = set(domains) - set(domain_values)
    unknown = set(domain_values) - set(domains)
    if missing or unknown:
        raise ValueError(f"Domain values mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if image_counts is not None:
        for domain in domains:
            count = int(image_counts[domain])
            if count < 0:
                raise ValueError(f"Validation image count for {domain} must be >= 0")
            if count == 0 and validated_weights[domain] > 0:
                raise ValueError(f"Validation domain {domain!r} has zero images but nonzero weight")
    values = {domain: float(domain_values[domain]) for domain in domains}
    invalid = [domain for domain, value in values.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"Validation values must be finite: {invalid}")
    return sum(validated_weights[domain] * values[domain] for domain in domains)


def select_validation_value(raw_value: float, weighted_value: float, enabled: bool) -> float:
    """Keep disabled experiments on their exact existing aggregate metric."""
    return float(weighted_value if enabled else raw_value)


def is_better_validation_value(current: float, best: float, maximize: bool = True) -> bool:
    return current > best if maximize else current < best
