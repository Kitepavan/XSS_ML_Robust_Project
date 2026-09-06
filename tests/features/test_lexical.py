"""Focused tests for the deterministic lexical/handcrafted feature extractor (TDD).

The extractor treats payloads as inert text (never executed) and must work
without any fitted detector or browser validation.
"""

from __future__ import annotations


def _payloads() -> list[str]:
    return [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "Hello world, this is a normal comment",
        "<svg onload=alert(1)>",
        "Search results for cats and dogs",
        "%3Cscript%3Ealert(1)%3C%2Fscript%3E",
        "&#60;script&#62;alert(1)",
        "<a href=javascript:alert(1)>click</a>",
        "  spaced\tout\npayload  ",
        "<ScRiPt>alert(String.fromCharCode(88))</ScRiPt>",
    ]


class TestFeatureSchema:
    def test_feature_names_nonempty_unique_strings(self):
        from xssharden.features.lexical import FEATURE_NAMES

        assert len(FEATURE_NAMES) > 0
        assert all(isinstance(name, str) and name for name in FEATURE_NAMES)
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_expected_feature_families_present(self):
        from xssharden.features.lexical import FEATURE_NAMES

        names = set(FEATURE_NAMES)
        # Length / structural counts.
        assert "length" in names
        # Tag markers.
        assert any("tag" in n for n in names)
        # Event-handler markers.
        assert any("event" in n for n in names)
        # Script/javascript markers.
        assert any("script" in n for n in names)
        assert any("javascript" in n for n in names)
        # Encoding markers.
        assert any("encod" in n or "entity" in n or "percent" in n for n in names)
        # Punctuation/quote counts.
        assert any("quote" in n for n in names)
        # Whitespace.
        assert any("whitespace" in n or "space" in n for n in names)
        # Entropy or unique-character ratio.
        assert any("entropy" in n or "unique" in n for n in names)

    def test_matrix_width_matches_feature_names(self):
        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        matrix = featurize_lexical(_payloads())
        assert matrix.shape == (len(_payloads()), len(FEATURE_NAMES))


class TestDeterminism:
    def test_same_payload_same_row_twice(self):
        import numpy as np

        from xssharden.features.lexical import featurize_lexical

        first = featurize_lexical(_payloads())
        second = featurize_lexical(_payloads())
        np.testing.assert_array_equal(first, second)

    def test_single_vs_batch_row_equal(self):
        import numpy as np

        from xssharden.features.lexical import featurize_lexical

        batch = featurize_lexical(_payloads())
        for i, payload in enumerate(_payloads()):
            single = featurize_lexical([payload])
            np.testing.assert_array_equal(single[0], batch[i])

    def test_input_order_preserved(self):
        import numpy as np

        from xssharden.features.lexical import featurize_lexical

        payloads = _payloads()
        matrix = featurize_lexical(payloads)
        reversed_matrix = featurize_lexical(list(reversed(payloads)))
        np.testing.assert_array_equal(matrix[0], reversed_matrix[-1])

    def test_reordered_input_reorders_rows_only(self):
        import numpy as np

        from xssharden.features.lexical import featurize_lexical

        payloads = _payloads()
        matrix = featurize_lexical(payloads)
        other = featurize_lexical([payloads[1], payloads[0]])
        np.testing.assert_array_equal(other[0], matrix[1])
        np.testing.assert_array_equal(other[1], matrix[0])


class TestRecordAndStringInputs:
    def test_accepts_plain_strings(self):
        import numpy as np

        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        matrix = featurize_lexical(["<script>alert(1)</script>", "hello"])
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, len(FEATURE_NAMES))

    def test_accepts_record_dicts_consistent_with_strings(self):
        import numpy as np

        from xssharden.features.lexical import featurize_lexical

        payloads = _payloads()
        from_strings = featurize_lexical(payloads)
        records = [
            {"sample_id": f"s-{i}", "payload": payload} for i, payload in enumerate(payloads)
        ]
        from_records = featurize_lexical(records)
        np.testing.assert_array_equal(from_strings, from_records)

    def test_record_missing_payload_raises(self):
        import pytest

        from xssharden.features.lexical import featurize_lexical

        with pytest.raises((TypeError, ValueError)):
            featurize_lexical([{"sample_id": "s-1"}])

    def test_empty_input_raises(self):
        import pytest

        from xssharden.features.lexical import featurize_lexical

        with pytest.raises(ValueError, match="non-empty"):
            featurize_lexical([])

    def test_non_string_payload_raises(self):
        import pytest

        from xssharden.features.lexical import featurize_lexical

        with pytest.raises(TypeError):
            featurize_lexical(["ok", 123])

    def test_does_not_mutate_input_records(self):
        from xssharden.features.lexical import featurize_lexical

        records = [{"sample_id": "s-1", "payload": "<script>alert(1)</script>"}]
        snapshot = [dict(r) for r in records]
        featurize_lexical(records)
        assert records == snapshot


class TestInertStrings:
    def test_malicious_looking_strings_are_plain_counts(self):
        import numpy as np

        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        matrix = featurize_lexical(
            ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"]
        )
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        length_idx = list(FEATURE_NAMES).index("length")
        assert matrix[0, length_idx] == len("<script>alert(1)</script>")

    def test_no_execution_primitives_in_source(self):
        import xssharden.features.lexical as mod

        src = open(mod.__file__, encoding="utf-8").read().lower()
        for banned in ("os.system", "subprocess", "os.popen", "eval(", "exec("):
            assert banned not in src

    def test_no_browser_or_validator_imports_in_source(self):
        import ast

        import xssharden.features.lexical as mod

        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append((node.module or "").split(".")[0])
        for banned in ("playwright", "selenium"):
            assert banned not in imported
        assert "xssharden.validation" not in open(
            mod.__file__, encoding="utf-8"
        ).read()

    def test_no_detector_import_required(self):
        import ast

        # Feature extraction must not import model libraries at module import.
        import xssharden.features.lexical as mod

        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append((node.module or "").split(".")[0])
        for banned in ("xgboost", "lightgbm", "sklearn"):
            assert banned not in imported


class TestSpotChecks:
    def test_length_and_brackets(self):
        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        matrix = featurize_lexical(["<b>hi</b>"])
        names = list(FEATURE_NAMES)
        row = dict(zip(names, matrix[0].tolist()))
        assert row["length"] == len("<b>hi</b>")
        assert row["n_lt"] == "<b>hi</b>".count("<")
        assert row["n_gt"] == "<b>hi</b>".count(">")

    def test_script_and_event_markers_fire(self):
        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        matrix = featurize_lexical(["<img src=x onerror=alert(1)>"])
        names = list(FEATURE_NAMES)
        row = dict(zip(names, matrix[0].tolist()))
        assert row["n_script"] == 0
        assert row["n_event_handlers"] >= 1
        matrix2 = featurize_lexical(["<SCRIPT>alert(1)</SCRIPT>"])
        row2 = dict(zip(names, matrix2[0].tolist()))
        assert row2["n_script"] >= 1

    def test_javascript_and_encoding_markers(self):
        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        names = list(FEATURE_NAMES)

        row = dict(
            zip(
                names,
                featurize_lexical(["<a href=javascript:alert(1)>x</a>"])[0].tolist(),
            )
        )
        assert row["n_javascript"] >= 1

        row_pct = dict(
            zip(names, featurize_lexical(["%3Cscript%3E"])[0].tolist())
        )
        assert row_pct["n_pct"] == "%3Cscript%3E".count("%")
        assert row_pct["n_pct_encoded"] >= 1

        row_ent = dict(
            zip(names, featurize_lexical(["&#60;script"])[0].tolist())
        )
        assert row_ent["n_html_entity"] >= 1

    def test_quotes_whitespace_entropy_ranges(self):
        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        names = list(FEATURE_NAMES)
        payload = "a'b\"c`d(e);f=g  h"
        row = dict(zip(names, featurize_lexical([payload])[0].tolist()))
        assert row["n_single_quote"] == 1
        assert row["n_double_quote"] == 1
        assert row["n_backtick"] == 1
        assert row["n_whitespace"] == sum(c.isspace() for c in payload)
        assert 0.0 <= row["whitespace_ratio"] <= 1.0
        assert 0.0 < row["unique_char_ratio"] <= 1.0
        assert row["shannon_entropy"] >= 0.0

    def test_benign_plain_text_has_no_markers(self):
        from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

        names = list(FEATURE_NAMES)
        row = dict(
            zip(names, featurize_lexical(["hello world plain text"])[0].tolist())
        )
        assert row["n_lt"] == 0
        assert row["n_script"] == 0
        assert row["n_javascript"] == 0
        assert row["n_event_handlers"] == 0
