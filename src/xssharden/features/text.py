"""Shared character-level TF-IDF feature pipeline for XSSHarden.

This is the single shared ``featurize()`` implementation used identically
by every detector, so downstream comparisons cannot be invalidated by
divergent feature extraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Fixed, deterministic character n-gram configuration shared by all callers.
ANALYZER = "char"
NGRAM_RANGE: tuple[int, int] = (3, 5)
LOWERCASE = False


def _build_vectorizer() -> TfidfVectorizer:
    """Return a fresh vectorizer with the shared deterministic settings."""
    return TfidfVectorizer(
        analyzer=ANALYZER,
        ngram_range=NGRAM_RANGE,
        lowercase=LOWERCASE,
    )


def featurize(
    payloads: Sequence[str],
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[csr_matrix | Any, TfidfVectorizer]:
    """Featurize XSS payloads with character-level TF-IDF.

    Parameters
    ----------
    payloads:
        Non-empty sequence of raw payload strings.
    vectorizer:
        Optional fitted vectorizer for reuse. When ``None`` (the default),
        a new vectorizer is built with the shared settings and fitted on
        *payloads*. When given, *payloads* are transformed with it without
        refitting, so the vocabulary is preserved.

    Returns
    -------
    tuple
        ``(matrix, vectorizer)`` where ``matrix`` is the TF-IDF feature
        matrix with one row per payload and ``vectorizer`` is the fitted
        vectorizer (the passed-in instance when reusing, otherwise new).

    Raises
    ------
    ValueError
        If *payloads* is empty.
    TypeError
        If any element of *payloads* is not a string.
    """
    if payloads is None or len(payloads) == 0:
        raise ValueError("payloads must be a non-empty sequence of strings")
    for index, payload in enumerate(payloads):
        if not isinstance(payload, str):
            raise TypeError(
                f"payload at index {index} must be str, "
                f"got {type(payload).__name__}"
            )

    if vectorizer is None:
        fitted = _build_vectorizer()
        matrix = fitted.fit_transform(list(payloads))
        return matrix, fitted

    matrix = vectorizer.transform(list(payloads))
    return matrix, vectorizer
