from __future__ import annotations

"""Inter-rater agreement primitives, implemented without external dependencies.

The test run treats the three models as three raters coding the same abstracts.
For each coded field these functions quantify how much the raters agree:

* ``percent_agreement`` / ``mean_pairwise_percent_agreement`` (observed agreement),
* ``cohen_kappa`` (chance-corrected, two raters),
* ``fleiss_kappa`` (chance-corrected, many raters, one metric per field),
* ``krippendorff_alpha`` (nominal; robust reliability coefficient).

All inputs are sequences of categorical labels, aligned by item. Every model
codes every abstract, so there is no missing data, but the implementations
tolerate it where noted.
"""

import itertools
import math
from collections import Counter

import numpy as np


def percent_agreement(a: list, b: list) -> float:
    """Fraction of items where two raters give the same label."""
    pairs = [(x, y) for x, y in zip(a, b) if x != "" and y != ""]
    if not pairs:
        return float("nan")
    same = sum(1 for x, y in pairs if x == y)
    return same / len(pairs)


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters over categorical labels."""
    pairs = [(x, y) for x, y in zip(a, b) if x != "" and y != ""]
    n = len(pairs)
    if n == 0:
        return float("nan")
    categories = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = np.zeros((k, k))
    for x, y in pairs:
        matrix[index[x], index[y]] += 1
    po = np.trace(matrix) / n
    row = matrix.sum(axis=1) / n
    col = matrix.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else float("nan")
    return (po - pe) / (1 - pe)


def mean_pairwise_percent_agreement(columns: list[list]) -> float:
    """Average observed agreement across all rater pairs."""
    values = [percent_agreement(a, b) for a, b in itertools.combinations(columns, 2)]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def mean_pairwise_cohen_kappa(columns: list[list]) -> float:
    """Average Cohen's kappa across all rater pairs."""
    values = [cohen_kappa(a, b) for a, b in itertools.combinations(columns, 2)]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def unanimous_rate(columns: list[list]) -> float:
    """Fraction of items on which all raters agree."""
    n_items = len(columns[0])
    if n_items == 0:
        return float("nan")
    agree = 0
    for i in range(n_items):
        labels = [col[i] for col in columns if col[i] != ""]
        if labels and len(set(labels)) == 1:
            agree += 1
    return agree / n_items


def fleiss_kappa(columns: list[list]) -> float:
    """Fleiss' kappa for a fixed number of raters over many items.

    ``columns`` is a list of rater columns (each a list of labels aligned by
    item). Items are the abstracts; the raters are the models.
    """
    n_raters = len(columns)
    n_items = len(columns[0])
    if n_raters < 2 or n_items == 0:
        return float("nan")

    categories = sorted({label for col in columns for label in col if label != ""})
    cat_index = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((n_items, len(categories)))
    valid_items = []
    for i in range(n_items):
        row_labels = [columns[r][i] for r in range(n_raters) if columns[r][i] != ""]
        if len(row_labels) < 2:
            continue
        valid_items.append(i)
        for label in row_labels:
            counts[i, cat_index[label]] += 1

    counts = counts[valid_items]
    if counts.size == 0:
        return float("nan")

    n_ij_rowsum = counts.sum(axis=1)  # raters per item (constant here)
    # Agreement per item.
    p_i = (np.sum(counts * counts, axis=1) - n_ij_rowsum) / (n_ij_rowsum * (n_ij_rowsum - 1))
    p_bar = float(np.mean(p_i))
    # Category marginals.
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = float(np.sum(p_j * p_j))
    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else float("nan")
    return (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha(columns: list[list]) -> float:
    """Krippendorff's alpha with the nominal difference metric.

    ``columns`` is a list of rater columns aligned by item. Missing values may
    be encoded as empty strings; units with fewer than two values are skipped.
    """
    n_raters = len(columns)
    n_items = len(columns[0])
    if n_raters < 2 or n_items == 0:
        return float("nan")

    categories = sorted({label for col in columns for label in col if label != ""})
    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    if k <= 1:
        # Only one category observed: perfect agreement by definition.
        return 1.0

    coincidence = np.zeros((k, k))
    for i in range(n_items):
        labels = [columns[r][i] for r in range(n_raters) if columns[r][i] != ""]
        m_u = len(labels)
        if m_u < 2:
            continue
        counts = Counter(labels)
        for c_label, c_n in counts.items():
            ci = cat_index[c_label]
            for k_label, k_n in counts.items():
                kj = cat_index[k_label]
                if ci == kj:
                    coincidence[ci, kj] += c_n * (c_n - 1) / (m_u - 1)
                else:
                    coincidence[ci, kj] += c_n * k_n / (m_u - 1)

    n_c = coincidence.sum(axis=1)
    n_total = n_c.sum()
    if n_total <= 1:
        return float("nan")

    # Nominal metric. With D_o = observed_offdiag / n_total and
    # D_e = (n_total^2 - sum n_c^2) / (n_total (n_total - 1)),
    # alpha = 1 - D_o / D_e simplifies to the closed form below.
    observed_offdiag = n_total - np.trace(coincidence)
    expected_offdiag = n_total * n_total - float(np.sum(n_c * n_c))
    if expected_offdiag <= 0:
        return 1.0 if observed_offdiag == 0 else float("nan")
    alpha = 1 - (n_total - 1) * observed_offdiag / expected_offdiag
    return float(alpha)


# Landis and Koch (1977) qualitative bands for kappa-like coefficients.
LANDIS_KOCH_BANDS = [
    (0.0, "Poor"),
    (0.20, "Slight"),
    (0.40, "Fair"),
    (0.60, "Moderate"),
    (0.80, "Substantial"),
    (1.01, "Almost perfect"),
]


def landis_koch_label(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if value < 0:
        return "Poor"
    for upper, label in LANDIS_KOCH_BANDS[1:]:
        if value < upper:
            return label
    return "Almost perfect"
