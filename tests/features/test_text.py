"""Focused tests for the shared character-level TF-IDF pipeline (Task 4).

Covers the single shared ``featurize()`` entry point used identically by
every detector: deterministic character n-grams, fit/transform reuse via a
reusable fitted vectorizer, and clear empty-input handling.
"""

from __future__ import annotations


PAYLOADS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "Hello, world!",
]

XSS_PAYLOADS = [
    "<script>alert('xss')</script>",
    "<ScRiPt>alert(1)</ScRiPt>",
    "<svg onload=alert(1)>",
]


# ===================================================================
# Section 1 – Deterministic character n-gram features
# ===================================================================

class TestDeterministicCharNgrams:
    """featurize() must build deterministic character n-gram TF-IDF features."""

    def test_returns_matrix_and_vectorizer(self):
        from xssharden.features.text import featurize

        matrix, vectorizer = featurize(PAYLOADS)
        assert matrix.shape[0] == len(PAYLOADS)
        assert matrix.shape[1] > 0
        assert vectorizer is not None

    def test_uses_character_analyzer(self):
        from xssharden.features.text import featurize

        _, vectorizer = featurize(PAYLOADS)
        assert vectorizer.analyzer == "char"

    def test_uses_fixed_ngram_range(self):
        from xssharden.features.text import featurize

        _, first = featurize(PAYLOADS)
        _, second = featurize(XSS_PAYLOADS)
        assert isinstance(first.ngram_range, tuple)
        assert first.ngram_range == second.ngram_range

    def test_deterministic_across_fresh_fits(self):
        from xssharden.features.text import featurize

        matrix_a, vec_a = featurize(PAYLOADS)
        matrix_b, vec_b = featurize(PAYLOADS)
        assert vec_a.vocabulary_ == vec_b.vocabulary_
        assert (matrix_a != matrix_b).nnz == 0

    def test_character_ngrams_capture_substrings(self):
        from xssharden.features.text import featurize

        _, vectorizer = featurize(["<script>alert(1)</script>"])
        vocab = vectorizer.vocabulary_
        assert any("scr" in term for term in vocab), (
            "expected a character n-gram containing 'scr'"
        )


# ===================================================================
# Section 2 – Fit/transform reuse
# ===================================================================

class TestFitTransformReuse:
    """Passing a fitted vectorizer must transform without refitting."""

    def test_reuses_fitted_vectorizer_identity(self):
        from xssharden.features.text import featurize

        _, fitted = featurize(PAYLOADS)
        _, reused = featurize(["<script>alert(2)</script>"], vectorizer=fitted)
        assert reused is fitted

    def test_reuse_preserves_vocabulary(self):
        from xssharden.features.text import featurize

        _, fitted = featurize(PAYLOADS)
        before = dict(fitted.vocabulary_)
        featurize(["completely unseen payload %$#@!"], vectorizer=fitted)
        assert fitted.vocabulary_ == before

    def test_reuse_output_width_matches_fitted_features(self):
        from xssharden.features.text import featurize

        fit_matrix, fitted = featurize(PAYLOADS)
        transform_matrix, _ = featurize(XSS_PAYLOADS, vectorizer=fitted)
        assert transform_matrix.shape[0] == len(XSS_PAYLOADS)
        assert transform_matrix.shape[1] == fit_matrix.shape[1]

    def test_reuse_matches_direct_transform(self):
        from xssharden.features.text import featurize

        _, fitted = featurize(PAYLOADS)
        via_featurize, _ = featurize(XSS_PAYLOADS, vectorizer=fitted)
        via_direct = fitted.transform(XSS_PAYLOADS)
        assert (via_featurize != via_direct).nnz == 0


# ===================================================================
# Section 3 – Empty-input handling
# ===================================================================

class TestEmptyInputHandling:
    """Empty payload lists must fail loudly, in both fit and reuse modes."""

    def test_empty_list_raises_value_error(self):
        from xssharden.features.text import featurize

        import pytest

        with pytest.raises(ValueError, match="non-empty"):
            featurize([])

    def test_empty_list_with_vectorizer_raises_value_error(self):
        from xssharden.features.text import featurize

        import pytest

        _, fitted = featurize(PAYLOADS)
        with pytest.raises(ValueError, match="non-empty"):
            featurize([], vectorizer=fitted)

    def test_row_count_matches_input(self):
        from xssharden.features.text import featurize

        matrix, _ = featurize(["<script>alert(1)</script>"])
        assert matrix.shape[0] == 1
