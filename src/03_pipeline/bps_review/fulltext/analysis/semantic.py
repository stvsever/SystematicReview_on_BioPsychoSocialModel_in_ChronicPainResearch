from __future__ import annotations

"""Semantic overlap of the open extraction lists (scheme 3).

The lexical Jaccard in ``reliability.py`` asks whether two coders wrote the same
string. That is the wrong question for this review. ``pain catastrophising`` and
``catastrophic thinking about pain`` are the same construct, ``social support``
and ``perceived social support`` are the same factor, ``fear-avoidance model``
and ``fear avoidance beliefs model`` are the same framework, and a string
comparison scores all of those as a disagreement. The measured overlap is then a
property of the wording rather than of the reading, which is exactly the
confusion this review exists to remove.

This module embeds every normalized label once with a sentence-embedding model,
and counts two labels as the same concept when their cosine similarity passes a
threshold. Overlap becomes a soft Jaccard over a one-to-one matching between the
two label sets:

    overlap = matched / (len(a) + len(b) - matched)

which reduces to the lexical Jaccard when the threshold is 1.0, so the two
numbers stay on one scale and can be reported side by side.

The threshold is deliberately permissive rather than strict. A strict threshold
would only merge near-identical strings and would reproduce the lexical result;
the point is to merge different wordings of one concept while keeping genuinely
different concepts apart. ``SIMILARITY_THRESHOLD`` is the single place where that
choice is made, and the similarity distribution of the run is written next to the
result so the choice stays inspectable.

Embeddings are cached on disk and keyed by the label text, so a rerun of the
analysis costs nothing and the numbers are reproducible without a network.

Writes, next to the lexical tables under ``03_reliability``:
``14_semantic_extraction_overlap.csv`` and ``semantic_overlap_summary.json``.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bps_review.fulltext.analysis.spaces import space_labels
from bps_review.fulltext.config import (
    EXTRACTION_SPACES,
    MODEL_LABELS,
    ExtractionSpace,
    reliability_dir,
)
from bps_review.llm.openrouter import SEMANTIC_EMBEDDING_MODEL, embed_texts_concurrent
from bps_review.utils.io import ensure_parent, write_csv, write_json
from bps_review.utils.paths import project_path


# Cosine similarity at which two labels are treated as the same concept.
# Calibrated by reading the matched pairs band by band: at 0.65 the matches are
# wording variants of one thing (pain catastrophising / catastrophic thinking,
# kinesiophobia / fear of movement, social support / perceived social support),
# while neighbouring but distinct constructs stay apart. Below 0.60 the band
# starts merging genuinely different factors, which would inflate agreement, so
# the threshold errs on the conservative side of "mild".
SIMILARITY_THRESHOLD = 0.65

# Reported next to the result, so the reader can see that no conclusion depends
# on where exactly the line is drawn.
SENSITIVITY_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)

# The space whose sensitivity is reported alongside the mean. The psychological
# concepts are the richest open vocabulary of the scheme and the one RQ3 rests on,
# so it is where a threshold choice would do the most damage.
SENSITIVITY_ANCHOR_SPACE = "psychological_concepts"

# Every comparable vocabulary of the scheme, not only the thirteen extraction
# lists: see ``config.EXTRACTION_SPACES``. Comparing lists by item identity asks
# whether two coders found the same things, and that is one question out of
# several the extraction supports. The constructs a coder says carry the
# biological domain, the measure a construct is operationalized with, the elements
# read into a definition of the model, the constructs a conceptual problem
# concerns: each is its own open vocabulary, and each is a place where two coders
# can read a paper the same way and write different words.
#
# The record-level free-text lists (conceptual tensions, additional observations)
# stay out of both metrics on purpose. They are sentences written for a human
# reader rather than labels, so a set comparison over them is not defined.
SEMANTIC_SPACES: list[ExtractionSpace] = list(EXTRACTION_SPACES)

# Kept as a name because it reads well in the summary and the notebook.
SEMANTIC_LIST_FIELDS: list[str] = [
    space.field for space in SEMANTIC_SPACES if space.layer == "identity"
]


def embedding_store_dir() -> Path:
    """Local vector store. Ignored by Git: it is a cache, not a review artifact."""
    slug = SEMANTIC_EMBEDDING_MODEL.replace("/", "_").replace("-", "_").replace(".", "_")
    return project_path("data", "interim", "embeddings", slug)


class LabelEmbeddingStore:
    """A tiny on-disk vector store keyed by the label text.

    One matrix and one index file, both readable without this package. Only
    labels that are not already present are sent to the provider, so the first
    run pays for the vocabulary and every later run is free.
    """

    def __init__(self, directory: Path | None = None, model: str = SEMANTIC_EMBEDDING_MODEL):
        self.dir = Path(directory) if directory is not None else embedding_store_dir()
        self.model = model
        self.labels: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.usage: dict = {}
        self._position: dict[str, int] = {}
        self._load()

    # ---------------------------------------------------------------- storage
    @property
    def index_path(self) -> Path:
        return self.dir / "index.json"

    @property
    def vectors_path(self) -> Path:
        return self.dir / "vectors.npy"

    def _load(self) -> None:
        if not (self.index_path.exists() and self.vectors_path.exists()):
            return
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if index.get("model") != self.model:
            return
        matrix = np.load(self.vectors_path)
        labels = list(index.get("labels", []))
        if len(labels) != matrix.shape[0]:
            return
        self.labels = labels
        self.matrix = matrix.astype(np.float32, copy=False)
        self._position = {label: row for row, label in enumerate(self.labels)}

    def save(self) -> None:
        ensure_parent(self.vectors_path)
        np.save(self.vectors_path, self.matrix)
        self.index_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "dimensions": int(self.matrix.shape[1]) if self.matrix.size else 0,
                    "n_labels": len(self.labels),
                    "labels": self.labels,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    # -------------------------------------------------------------- embedding
    def ensure(self, texts: list[str], max_workers: int = 8, batch_size: int = 96) -> dict:
        """Embed every text that is not cached yet. Returns the usage of this call."""
        missing = [text for text in dict.fromkeys(texts) if text and text not in self._position]
        if not missing:
            return {}
        vectors, usage = embed_texts_concurrent(
            missing, model=self.model, batch_size=batch_size, max_workers=max_workers
        )
        block = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        block = block / np.clip(norms, 1e-12, None)
        if self.matrix.size:
            self.matrix = np.vstack([self.matrix, block])
        else:
            self.matrix = block
        for text in missing:
            self._position[text] = len(self.labels)
            self.labels.append(text)
        self.save()
        self.usage = usage
        return usage

    def vectors_for(self, texts: list[str]) -> np.ndarray:
        rows = [self._position[text] for text in texts if text in self._position]
        if not rows:
            return np.zeros((0, self.matrix.shape[1] if self.matrix.size else 0), dtype=np.float32)
        return self.matrix[rows]

    def known(self, texts: list[str]) -> list[str]:
        return [text for text in texts if text in self._position]


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def _similarity(first: list[str], second: list[str], store: LabelEmbeddingStore) -> np.ndarray:
    left = store.vectors_for(first)
    right = store.vectors_for(second)
    if left.size == 0 or right.size == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)
    return left @ right.T


def greedy_match(similarity: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one matching, strongest pair first.

    Greedy is the right level of machinery here: the label sets hold at most a
    dozen items, matches above the threshold are rarely contested, and a greedy
    pass is deterministic and explainable in one sentence.
    """
    if similarity.size == 0:
        return []
    pairs = [
        (int(i), int(j), float(similarity[i, j]))
        for i in range(similarity.shape[0])
        for j in range(similarity.shape[1])
        if similarity[i, j] >= threshold
    ]
    pairs.sort(key=lambda item: -item[2])
    used_left: set[int] = set()
    used_right: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for i, j, score in pairs:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matched.append((i, j, score))
    return matched


def semantic_jaccard(first: list[str], second: list[str], store: LabelEmbeddingStore,
                     threshold: float) -> float:
    if not first and not second:
        return float("nan")
    matched = len(greedy_match(_similarity(first, second, store), threshold))
    union = len(first) + len(second) - matched
    if union <= 0:
        return float("nan")
    return matched / union


def _has_shared_concept(sets: list[list[str]], store: LabelEmbeddingStore,
                        threshold: float) -> bool:
    """True when at least one label of the first set has a match in every other set."""
    non_empty = [labels for labels in sets if labels]
    if len(non_empty) < 2:
        return False
    anchor, others = non_empty[0], non_empty[1:]
    similarities = [_similarity(anchor, other, store) for other in others]
    for index in range(len(anchor)):
        if all(sim.shape[1] and sim[index].max() >= threshold for sim in similarities):
            return True
    return False


def cluster_labels(labels: list[str], store: LabelEmbeddingStore, threshold: float) -> int:
    """How many distinct concepts a label vocabulary collapses into."""
    unique = store.known(sorted(set(labels)))
    if not unique:
        return 0
    vectors = store.vectors_for(unique)
    similarity = vectors @ vectors.T
    assigned: dict[int, int] = {}
    clusters = 0
    for index in range(len(unique)):
        if index in assigned:
            continue
        assigned[index] = clusters
        for other in range(index + 1, len(unique)):
            if other not in assigned and similarity[index, other] >= threshold:
                assigned[other] = clusters
        clusters += 1
    return clusters


# --------------------------------------------------------------------------
# Per-space computation
# --------------------------------------------------------------------------
def present_spaces(long_df: pd.DataFrame,
                   spaces: list[ExtractionSpace] | None = None) -> list[ExtractionSpace]:
    """The spaces this run can actually answer: source list present and non-empty.

    A space whose extraction list the run does not carry, or which no coder ever
    filled, is dropped rather than reported as an empty row. An older run then
    describes exactly what it holds, while the registry keeps describing the whole
    scheme.
    """
    available = []
    for space in spaces if spaces is not None else SEMANTIC_SPACES:
        if space.field not in long_df.columns:
            continue
        if any(space_labels(value, space) for value in long_df[space.field].tolist()):
            available.append(space)
    return available


def _label_lists(long_df: pd.DataFrame,
                 space: ExtractionSpace) -> tuple[dict[str, list[list[str]]], list[str]]:
    pivot = long_df.pivot(index="record_id", columns="model_label",
                          values=space.field).reindex(columns=MODEL_LABELS)
    return {
        model: [sorted(space_labels(value, space)) for value in pivot[model].tolist()]
        for model in MODEL_LABELS
    }, list(pivot.index)


def compute_semantic_overlap(
    long_df: pd.DataFrame,
    store: LabelEmbeddingStore,
    threshold: float = SIMILARITY_THRESHOLD,
    spaces: list[ExtractionSpace] | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    """Per comparison space: soft Jaccard over embedded labels, next to the lexical one.

    Both columns are computed here, from the same label sets, by the same pass.
    Reading a semantic score against a lexical one produced elsewhere would
    compare two instruments rather than two ways of measuring one thing.
    """
    rows = []
    best_matches: list[float] = []
    for space in present_spaces(long_df, spaces):
        per_model, record_ids = _label_lists(long_df, space)
        semantic_scores: list[float] = []
        lexical_scores: list[float] = []
        for first, second in itertools.combinations(MODEL_LABELS, 2):
            for a, b in zip(per_model[first], per_model[second]):
                if not a and not b:
                    continue
                similarity = _similarity(a, b, store)
                if similarity.size:
                    best_matches.extend(similarity.max(axis=1).tolist())
                matched = len(greedy_match(similarity, threshold))
                union = len(a) + len(b) - matched
                if union > 0:
                    semantic_scores.append(matched / union)
                lexical_union = len(set(a) | set(b))
                if lexical_union:
                    lexical_scores.append(len(set(a) & set(b)) / lexical_union)

        shared = sum(
            1
            for index in range(len(record_ids))
            if _has_shared_concept([per_model[model][index] for model in MODEL_LABELS], store,
                                   threshold)
        )
        all_labels = [label for model in MODEL_LABELS for labels in per_model[model]
                      for label in labels]
        per_paper = [len(per_model[model][index]) for model in MODEL_LABELS
                     for index in range(len(record_ids))]
        n_distinct = len(set(all_labels))
        n_concepts = cluster_labels(all_labels, store, threshold)
        lexical_mean = float(np.mean(lexical_scores)) if lexical_scores else float("nan")
        semantic_mean = float(np.mean(semantic_scores)) if semantic_scores else float("nan")
        rows.append(
            {
                "space": space.name,
                "space_label": space.label,
                "layer": space.layer,
                "label_kind": space.kind,
                "source_field": space.field,
                "read_from": space.sublist or " + ".join(space.keys),
                "question": space.question,
                "mean_pairwise_jaccard": lexical_mean,
                "mean_pairwise_semantic_jaccard": semantic_mean,
                "median_pairwise_semantic_jaccard": float(np.median(semantic_scores))
                if semantic_scores else float("nan"),
                "semantic_gain": semantic_mean - lexical_mean,
                "share_papers_with_shared_concept": shared / len(record_ids)
                if record_ids else float("nan"),
                "n_comparable_pairs": len(semantic_scores),
                "mean_labels_per_coding": round(float(np.mean(per_paper)), 2) if per_paper else 0.0,
                "n_label_entries": len(all_labels),
                "n_distinct_labels": n_distinct,
                "n_semantic_concepts": n_concepts,
                "label_inflation": (n_distinct / n_concepts) if n_concepts else float("nan"),
            }
        )
    return pd.DataFrame(rows), best_matches


def duplicate_pair_overlap(
    long_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    store: LabelEmbeddingStore,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """Same model, same text, twice: the reproducibility ceiling of an open label list.

    When the corpus contains a paper that is byte-identical to another paper,
    every model has coded the same text twice without knowing it. The overlap of
    those two codings is the ceiling any cross-model comparison can reach.
    """
    if corpus_df is None or "duplicate_of" not in corpus_df.columns:
        return {}
    pairs = [
        (str(row["duplicate_of"]), str(row["record_id"]))
        for _, row in corpus_df.iterrows()
        if str(row.get("duplicate_of") or "").strip()
    ]
    if not pairs:
        return {}
    per_space: dict[str, list[float]] = {}
    for original, copy in pairs:
        for space in present_spaces(long_df):
            for model in MODEL_LABELS:
                left = long_df[(long_df.record_id == original) & (long_df.model_label == model)]
                right = long_df[(long_df.record_id == copy) & (long_df.model_label == model)]
                if left.empty or right.empty:
                    continue
                a = sorted(space_labels(left[space.field].iloc[0], space))
                b = sorted(space_labels(right[space.field].iloc[0], space))
                score = semantic_jaccard(a, b, store, threshold)
                if not np.isnan(score):
                    per_space.setdefault(space.name, []).append(score)
    if not per_space:
        return {}
    return {
        "pairs": [{"original": original, "duplicate": copy} for original, copy in pairs],
        "per_space": {name: round(float(np.mean(scores)), 3) for name, scores in per_space.items()},
        "mean_over_spaces": round(
            float(np.mean([np.mean(scores) for scores in per_space.values()])), 3),
    }


def build_semantic_overlap(
    long_df: pd.DataFrame,
    corpus_df: pd.DataFrame | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
    spaces: list[ExtractionSpace] | None = None,
    write: bool = True,
    out_dir: Path | None = None,
    store_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Embed the label vocabulary of a run and quantify overlap in every space."""
    store = LabelEmbeddingStore(store_dir)
    available = present_spaces(long_df, spaces)
    vocabulary = sorted(
        {
            label
            for space in available
            for value in long_df[space.field].tolist()
            for label in space_labels(value, space)
        }
    )
    cached = len(store.known(vocabulary))
    if verbose:
        print(f"  spaces: {len(available)} of {len(spaces or SEMANTIC_SPACES)} the scheme declares")
        print(f"  labels: {len(vocabulary)} ({cached} cached, {len(vocabulary) - cached} new)")
    usage = store.ensure(vocabulary)
    if verbose and usage:
        print(f"  embedded in {usage.get('requests', 0)} requests, "
              f"{usage.get('total_tokens', 0)} tokens, ${usage.get('cost_usd', 0):.4f}")

    overlap, best_matches = compute_semantic_overlap(long_df, store, threshold, available)
    duplicates = (
        duplicate_pair_overlap(long_df, corpus_df, store, threshold)
        if corpus_df is not None else {}
    )

    sensitivity = {}
    for candidate in SENSITIVITY_THRESHOLDS:
        frame, _ = compute_semantic_overlap(long_df, store, candidate, available)
        anchor = frame.loc[frame["space"] == SENSITIVITY_ANCHOR_SPACE,
                           "mean_pairwise_semantic_jaccard"]
        free_text = frame[frame["label_kind"] == "free text"]
        sensitivity[f"{candidate:.2f}"] = {
            "mean": round(float(frame["mean_pairwise_semantic_jaccard"].mean()), 3),
            "mean_free_text_spaces": round(
                float(free_text["mean_pairwise_semantic_jaccard"].mean()), 3) if len(free_text) else None,
            SENSITIVITY_ANCHOR_SPACE: round(float(anchor.iloc[0]), 3) if len(anchor) else None,
        }

    summary = {
        "embedding_model": store.model,
        "embedding_dimensions": int(store.matrix.shape[1]) if store.matrix.size else 0,
        "similarity_threshold": threshold,
        "n_spaces_declared": len(spaces or SEMANTIC_SPACES),
        "n_spaces_measured": len(available),
        "spaces_by_layer": {
            layer: int((overlap["layer"] == layer).sum()) for layer in ("identity", "vocabulary", "filtered")
        } if not overlap.empty else {},
        "n_labels_embedded": len(vocabulary),
        "n_labels_new_this_run": len(vocabulary) - cached,
        "embedding_usage": usage,
        "match_similarity_distribution": {
            "n_label_comparisons": len(best_matches),
            "median_best_match": round(float(np.median(best_matches)), 3) if best_matches else None,
            "share_above_threshold": round(
                float(np.mean([1 if value >= threshold else 0 for value in best_matches])), 3)
            if best_matches else None,
        },
        "mean_semantic_jaccard": round(
            float(overlap["mean_pairwise_semantic_jaccard"].mean()), 4) if not overlap.empty else None,
        "mean_lexical_jaccard": round(
            float(overlap["mean_pairwise_jaccard"].mean()), 4) if not overlap.empty else None,
        "mean_semantic_jaccard_free_text": round(
            float(overlap.loc[overlap["label_kind"] == "free text",
                              "mean_pairwise_semantic_jaccard"].mean()), 4) if not overlap.empty else None,
        "threshold_sensitivity": sensitivity,
        "per_space": {
            row["space"]: {
                "label": row["space_label"],
                "layer": row["layer"],
                "label_kind": row["label_kind"],
                "lexical": round(float(row["mean_pairwise_jaccard"]), 3),
                "semantic": round(float(row["mean_pairwise_semantic_jaccard"]), 3),
                "label_entries": int(row["n_label_entries"]),
                "distinct_labels": int(row["n_distinct_labels"]),
                "semantic_concepts": int(row["n_semantic_concepts"]),
            }
            for _, row in overlap.iterrows()
        },
        "duplicate_pair_overlap": duplicates,
    }

    if write:
        out = Path(out_dir) if out_dir is not None else reliability_dir()
        write_csv(out / "14_semantic_extraction_overlap.csv", overlap)
        write_json(out / "semantic_overlap_summary.json", summary)

    return {"overlap": overlap, "summary": summary, "store": store}
