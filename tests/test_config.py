"""Tests for experiment configuration and stable experiment IDs (Task 3)."""

from __future__ import annotations

import pytest

from xssharden.config import make_experiment_id


# ===================================================================
# Section 1 – Stable experiment IDs
# ===================================================================

class TestMakeExperimentId:
    """make_experiment_id must produce deterministic, stable hashes."""

    def test_returns_string(self):
        result = make_experiment_id({"model": "rf", "features": "char_tfidf"}, seed=42)
        assert isinstance(result, str)

    def test_non_empty(self):
        result = make_experiment_id({"model": "rf"}, seed=42)
        assert len(result) > 0

    def test_deterministic_same_input(self):
        config = {"model": "rf", "features": "char_tfidf", "ngram_range": [1, 3]}
        first = make_experiment_id(config, seed=42)
        second = make_experiment_id(config, seed=42)
        assert first == second

    def test_different_seed_different_id(self):
        config = {"model": "rf"}
        id_42 = make_experiment_id(config, seed=42)
        id_99 = make_experiment_id(config, seed=99)
        assert id_42 != id_99

    def test_different_config_different_id(self):
        id_a = make_experiment_id({"model": "rf"}, seed=42)
        id_b = make_experiment_id({"model": "svm"}, seed=42)
        assert id_a != id_b

    def test_key_order_independent(self):
        """Config dicts with the same keys/values in different order must
        produce the same experiment ID."""
        id_a = make_experiment_id({"model": "rf", "features": "char_tfidf"}, seed=42)
        id_b = make_experiment_id({"features": "char_tfidf", "model": "rf"}, seed=42)
        assert id_a == id_b

    def test_contains_hex_characters(self):
        """The ID should look like a hex hash (lowercase hex chars)."""
        result = make_experiment_id({"model": "rf"}, seed=42)
        import re
        assert re.fullmatch(r"[0-9a-f]+", result), (
            f"Expected hex string, got: {result}"
        )

    def test_empty_config(self):
        """Even an empty config should produce a valid ID."""
        result = make_experiment_id({}, seed=0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_nested_config(self):
        """Nested dicts/lists in config should be handled."""
        config = {
            "model": "rf",
            "params": {"n_estimators": 100, "max_depth": 10},
            "features": ["char_tfidf", "payload_length"],
        }
        result = make_experiment_id(config, seed=42)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stable_across_calls(self):
        """Calling 100 times with the same input always yields the same ID."""
        config = {"model": "rf", "seed": 42}
        results = {make_experiment_id(config, seed=42) for _ in range(100)}
        assert len(results) == 1
