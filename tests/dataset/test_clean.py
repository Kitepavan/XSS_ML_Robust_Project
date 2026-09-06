"""Tests for dataset schema validation, payload normalization, and deduplication (Task 2)."""

import pytest


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> dict:
    """Return a fully-formed record, allowing selective overrides."""
    base = {
        "sample_id": "S001",
        "payload": "<script>alert(1)</script>",
        "label": 1,
        "source": "github",
        "attack_category": "reflected_html",
        "split": "train",
    }
    base.update(overrides)
    return base


# ===================================================================
# Section 1 – Required-field validation (validate_records)
# ===================================================================

class TestValidateRecords:
    """validate_records must enforce required fields and types."""

    REQUIRED_FIELDS = ("sample_id", "payload", "label", "source", "attack_category", "split")

    # -- happy path --

    def test_valid_record_passes(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record()
        validated = validate_records([record])
        assert len(validated) == 1
        assert validated[0] == record

    def test_multiple_valid_records(self):
        from xssharden.dataset.schema import validate_records

        records = [_make_record(sample_id=f"S{i:03d}") for i in range(5)]
        validated = validate_records(records)
        assert len(validated) == 5

    # -- missing required fields --

    def test_rejects_record_missing_sample_id(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record()
        del record["sample_id"]
        with pytest.raises(ValueError, match="sample_id"):
            validate_records([record])

    def test_rejects_record_missing_payload(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record()
        del record["payload"]
        with pytest.raises(ValueError, match="payload"):
            validate_records([record])

    def test_rejects_record_missing_label(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record()
        del record["label"]
        with pytest.raises(ValueError, match="label"):
            validate_records([record])

    def test_rejects_record_missing_source(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record()
        del record["source"]
        with pytest.raises(ValueError, match="source"):
            validate_records([record])

    def test_rejects_record_missing_attack_category(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record()
        del record["attack_category"]
        with pytest.raises(ValueError, match="attack_category"):
            validate_records([record])

    def test_rejects_record_missing_split(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record()
        del record["split"]
        with pytest.raises(ValueError, match="split"):
            validate_records([record])

    def test_empty_payload_rejected(self):
        """Payload must be a non-empty string."""
        from xssharden.dataset.schema import validate_records

        record = _make_record(payload="")
        with pytest.raises(ValueError):
            validate_records([record])

    def test_whitespace_only_payload_rejected(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record(payload="   ")
        with pytest.raises(ValueError):
            validate_records([record])

    def test_empty_list_passes(self):
        from xssharden.dataset.schema import validate_records

        assert validate_records([]) == []

    # -- type checks --

    def test_rejects_non_dict_record(self):
        from xssharden.dataset.schema import validate_records

        with pytest.raises(TypeError):
            validate_records(["not a dict"])

    def test_label_must_be_int(self):
        from xssharden.dataset.schema import validate_records

        record = _make_record(label="malicious")
        with pytest.raises(TypeError, match="label"):
            validate_records([record])


# ===================================================================
# Section 2 – Stable payload normalization (normalize_payload)
# ===================================================================

class TestNormalizePayload:
    """normalize_payload must produce deterministic, stable output."""

    def test_basic_normalization(self):
        from xssharden.dataset.clean import normalize_payload

        result = normalize_payload("<script>alert(1)</script>")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_idempotent(self):
        """Normalizing twice must yield the same result."""
        from xssharden.dataset.clean import normalize_payload

        raw = "<Script > Alert(1) </Script>"
        first = normalize_payload(raw)
        second = normalize_payload(first)
        assert first == second

    def test_case_folding(self):
        """Normalisation must collapse case differences."""
        from xssharden.dataset.clean import normalize_payload

        lower = normalize_payload("<script>alert(1)</script>")
        upper = normalize_payload("<SCRIPT>ALERT(1)</SCRIPT>")
        assert lower == upper

    def test_whitespace_collapsing(self):
        """Extra internal whitespace must be collapsed."""
        from xssharden.dataset.clean import normalize_payload

        compact = normalize_payload("<script>alert(1)</script>")
        spaced = normalize_payload("<script>  alert(1)  </script>")
        assert compact == spaced

    def test_empty_string(self):
        from xssharden.dataset.clean import normalize_payload

        assert normalize_payload("") == ""

    def test_deterministic_across_calls(self):
        """Same input always yields the same output (no randomness)."""
        from xssharden.dataset.clean import normalize_payload

        payload = "javascript:alert(document.cookie)"
        results = [normalize_payload(payload) for _ in range(10)]
        assert len(set(results)) == 1


# ===================================================================
# Section 3 – Exact duplicate removal (deduplicate_records)
# ===================================================================

class TestDeduplicateExact:
    """deduplicate_records must remove byte-identical records."""

    def test_no_duplicates_unchanged(self):
        from xssharden.dataset.clean import deduplicate_records

        records = [_make_record(sample_id=f"S{i}") for i in range(3)]
        result = deduplicate_records(records, exact_only=True)
        assert len(result) == 3

    def test_exact_duplicates_removed(self):
        from xssharden.dataset.clean import deduplicate_records

        r = _make_record()
        records = [r, r.copy(), r.copy()]
        result = deduplicate_records(records, exact_only=True)
        assert len(result) == 1

    def test_preserves_first_occurrence(self):
        from xssharden.dataset.clean import deduplicate_records

        r = _make_record()
        records = [r, r.copy()]
        result = deduplicate_records(records, exact_only=True)
        assert result[0] is r or result[0] == r

    def test_different_records_kept(self):
        from xssharden.dataset.clean import deduplicate_records

        r1 = _make_record(sample_id="S1", payload="a")
        r2 = _make_record(sample_id="S2", payload="b")
        result = deduplicate_records([r1, r2], exact_only=True)
        assert len(result) == 2

    def test_empty_input(self):
        from xssharden.dataset.clean import deduplicate_records

        assert deduplicate_records([], exact_only=True) == []

    def test_single_record(self):
        from xssharden.dataset.clean import deduplicate_records

        r = [_make_record()]
        assert len(deduplicate_records(r, exact_only=True)) == 1


# ===================================================================
# Section 4 – Normalized duplicate removal
# ===================================================================

class TestDeduplicateNormalized:
    """deduplicate_records must also catch payload-equivalent duplicates."""

    def test_normalized_duplicates_removed(self):
        """Records differing only in payload case/whitespace collapse."""
        from xssharden.dataset.clean import deduplicate_records

        r1 = _make_record(payload="<script>alert(1)</script>")
        r2 = _make_record(payload="<Script>  Alert(1)  </Script>")
        result = deduplicate_records([r1, r2])
        assert len(result) == 1

    def test_distinct_payloads_kept(self):
        from xssharden.dataset.clean import deduplicate_records

        r1 = _make_record(payload="<script>alert(1)</script>")
        r2 = _make_record(payload="<script>alert(2)</script>")
        result = deduplicate_records([r1, r2])
        assert len(result) == 2

    def test_mixed_exact_and_normalized(self):
        from xssharden.dataset.clean import deduplicate_records

        base = _make_record(payload="<img onerror=alert(1)>")
        norm = _make_record(payload="<IMG  ONERROR=alert(1)>")
        diff = _make_record(payload="<img onerror=alert(2)>")
        result = deduplicate_records([base, norm, diff])
        assert len(result) == 2

    def test_normalized_dedup_idempotent(self):
        from xssharden.dataset.clean import deduplicate_records

        r1 = _make_record(payload="<Script>alert(1)</Script>")
        r2 = _make_record(payload="<script>alert(1)</script>")
        first = deduplicate_records([r1, r2])
        second = deduplicate_records(first)
        assert len(first) == len(second)

    def test_normalized_preserves_first_occurrence(self):
        from xssharden.dataset.clean import deduplicate_records

        r1 = _make_record(sample_id="KEEP", payload="<b> hi </b>")
        r2 = _make_record(sample_id="DROP", payload="<B>  HI  </B>")
        result = deduplicate_records([r1, r2])
        assert len(result) == 1
        assert result[0]["sample_id"] == "KEEP"
