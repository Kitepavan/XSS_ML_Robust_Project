"""Focused TDD tests for the provider-agnostic cloud-LLM variant adapter.

Conventions:

- Only synthetic inert strings are used as payloads. No downloaded dataset
  payloads are executed, no browser is launched, and no real network traffic
  is sent. Every transport is a fake injected by the test.
- Model output is treated as untrusted text: strict contract parsing,
  provenance checks, and deduplication are verified below.
"""

from __future__ import annotations

import json

import pytest


def _seed(**overrides):
    base = {
        "sample_id": "S001",
        "payload": '<script>alert("xss")</script>',
        "label": 1,
        "source": "train_corpus",
        "attack_category": "xss",
        "split": "train",
        "context_target": "reflected_html",
    }
    base.update(overrides)
    return base


def _api_response(variants):
    """Wrap variant dicts in an OpenAI-compatible chat-completion envelope."""
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [{"message": {"role": "assistant", "content": json.dumps({"variants": variants})}}],
    }


class _FakeClient:
    """Injectable stand-in for the OpenAI-compatible client (no network)."""

    def __init__(self, handler, model="fake-model"):
        self._handler = handler
        self._model = model
        self.calls = []

    @property
    def model(self):
        return self._model

    def chat(self, messages):
        self.calls.append(messages)
        return self._handler(messages)


def _ok_client(variants_per_seed=None, model="fake-model"):
    """Fake client echoing one valid variant per seed from the request."""
    def handler(messages):
        user_text = json.dumps(messages)
        seed_id = "S001"
        for message in messages:
            text = message.get("content", "")
            if '"seed_id"' in text:
                try:
                    start = text.index('"seed_id"')
                    _ = start
                except ValueError:
                    pass
        return _api_response(variants_per_seed if variants_per_seed is not None else [
            {
                "seed_id": seed_id,
                "payload": '<svg onload="prompt(1)">',
                "mutation_category": "tag_event_substitution",
                "context_target": "reflected_html",
            }
        ])
    return _FakeClient(handler, model=model)


# ---------------------------------------------------------------------------
# Configuration (explicit args / environment; no localhost default)
# ---------------------------------------------------------------------------


class TestLLMConfiguration:
    def test_config_requires_explicit_base_url_model_and_key(self):
        from xssharden.generation.llm import LLMConfig, LLMConfigurationError

        with pytest.raises(LLMConfigurationError):
            LLMConfig(base_url="", model="m", api_key="k")
        with pytest.raises(LLMConfigurationError):
            LLMConfig(base_url="https://api.example.com/v1", model="", api_key="k")
        with pytest.raises(LLMConfigurationError):
            LLMConfig(base_url="https://api.example.com/v1", model="m", api_key="")

    def test_config_rejects_non_http_base_url(self):
        from xssharden.generation.llm import LLMConfig, LLMConfigurationError

        with pytest.raises(LLMConfigurationError):
            LLMConfig(base_url="ftp://example.com/v1", model="m", api_key="k")

    def test_config_rejects_non_positive_timeout(self):
        from xssharden.generation.llm import LLMConfig, LLMConfigurationError

        with pytest.raises(LLMConfigurationError):
            LLMConfig(base_url="https://api.example.com/v1", model="m", api_key="k", timeout_s=0)
        with pytest.raises(LLMConfigurationError):
            LLMConfig(base_url="https://api.example.com/v1", model="m", api_key="k", timeout_s=-5)

    def test_from_env_reads_documented_variable_names(self, monkeypatch):
        from xssharden.generation.llm import LLMConfig

        monkeypatch.setenv("XSSHARDEN_LLM_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("XSSHARDEN_LLM_API_KEY", "env-key")
        monkeypatch.setenv("XSSHARDEN_LLM_MODEL", "env-model")
        monkeypatch.setenv("XSSHARDEN_LLM_TIMEOUT_S", "12.5")
        monkeypatch.setenv("XSSHARDEN_LLM_ORGANIZATION", "org-1")
        monkeypatch.setenv("XSSHARDEN_LLM_PROJECT", "proj-1")
        config = LLMConfig.from_env()
        assert config.base_url == "https://api.example.com/v1"
        assert config.api_key == "env-key"
        assert config.model == "env-model"
        assert config.timeout_s == pytest.approx(12.5)
        assert config.organization == "org-1"
        assert config.project == "proj-1"

    def test_from_env_missing_credentials_raise_without_network(self, monkeypatch):
        from xssharden.generation.llm import LLMConfig, LLMConfigurationError

        for var in (
            "XSSHARDEN_LLM_BASE_URL",
            "XSSHARDEN_LLM_API_KEY",
            "XSSHARDEN_LLM_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(LLMConfigurationError):
            LLMConfig.from_env()

    def test_from_env_has_no_localhost_default(self, monkeypatch):
        from xssharden.generation.llm import LLMConfig, LLMConfigurationError

        for var in (
            "XSSHARDEN_LLM_BASE_URL",
            "XSSHARDEN_LLM_API_KEY",
            "XSSHARDEN_LLM_MODEL",
            "XSSHARDEN_LLM_TIMEOUT_S",
            "XSSHARDEN_LLM_ORGANIZATION",
            "XSSHARDEN_LLM_PROJECT",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(LLMConfigurationError):
            LLMConfig.from_env()


# ---------------------------------------------------------------------------
# Auth headers and endpoint construction
# ---------------------------------------------------------------------------


class TestTransportHeadersAndEndpoint:
    def test_endpoint_appends_chat_completions_path(self):
        from xssharden.generation.llm import LLMConfig

        config = LLMConfig(
            base_url="https://api.example.com/v1/",
            model="m",
            api_key="k",
        )
        assert config.endpoint_url == "https://api.example.com/v1/chat/completions"

    def test_auth_headers_carry_bearer_key_and_json_content_type(self):
        from xssharden.generation.llm import LLMConfig

        config = LLMConfig(
            base_url="https://api.example.com/v1", model="m", api_key="secret-123"
        )
        headers = config.headers()
        assert headers["Authorization"] == "Bearer secret-123"
        assert headers["Content-Type"] == "application/json"
        assert "secret-123" not in json.dumps({k: v for k, v in headers.items() if k != "Authorization"})

    def test_optional_organization_and_project_headers(self):
        from xssharden.generation.llm import LLMConfig

        config = LLMConfig(
            base_url="https://api.example.com/v1",
            model="m",
            api_key="k",
            organization="org-1",
            project="proj-1",
        )
        headers = config.headers()
        assert headers["OpenAI-Organization"] == "org-1"
        assert headers["OpenAI-Project"] == "proj-1"

    def test_no_org_headers_by_default(self):
        from xssharden.generation.llm import LLMConfig

        config = LLMConfig(base_url="https://api.example.com/v1", model="m", api_key="k")
        headers = config.headers()
        assert "OpenAI-Organization" not in headers
        assert "OpenAI-Project" not in headers

    def test_client_posts_to_endpoint_with_auth_via_injected_transport(self):
        from xssharden.generation.llm import LLMConfig, OpenAICompatibleClient

        seen = {}

        def fake_transport(url, headers, body, timeout_s):
            seen["url"] = url
            seen["headers"] = headers
            seen["body"] = body
            seen["timeout_s"] = timeout_s
            return _api_response([])

        config = LLMConfig(
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key="key-abc",
            timeout_s=7.0,
        )
        client = OpenAICompatibleClient(config, transport=fake_transport)
        client.chat([{"role": "user", "content": "hello"}])
        assert seen["url"] == "https://api.example.com/v1/chat/completions"
        assert seen["headers"]["Authorization"] == "Bearer key-abc"
        assert seen["body"]["model"] == "test-model"
        assert seen["timeout_s"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Opt-in safety (default refuses before any network request)
# ---------------------------------------------------------------------------


class TestOptInSafety:
    def test_generation_refuses_without_explicit_opt_in(self):
        from xssharden.generation.llm import (
            LLMPayloadSubmissionNotAllowedError,
            generate_llm_variants,
        )

        client = _ok_client()
        with pytest.raises(LLMPayloadSubmissionNotAllowedError):
            generate_llm_variants([_seed()], client=client)
        assert client.calls == []

    def test_generation_refuses_when_opt_in_is_false(self):
        from xssharden.generation.llm import (
            LLMPayloadSubmissionNotAllowedError,
            generate_llm_variants,
        )

        client = _ok_client()
        with pytest.raises(LLMPayloadSubmissionNotAllowedError):
            generate_llm_variants([_seed()], client=client, allow_payload_submission=False)
        assert client.calls == []

    def test_opt_in_allows_exactly_one_request_per_seed(self):
        from xssharden.generation.llm import generate_llm_variants

        def handler(messages):
            return _api_response([
                {
                    "seed_id": "S001",
                    "payload": "<svg onload=prompt(1)>",
                    "mutation_category": "encoding",
                    "context_target": "reflected_html",
                }
            ])

        client = _FakeClient(handler)
        result = generate_llm_variants(
            [_seed(sample_id="S001")], client=client, allow_payload_submission=True
        )
        assert len(client.calls) == 1
        assert len(result.variants) == 1


# ---------------------------------------------------------------------------
# Strict response parsing and provenance
# ---------------------------------------------------------------------------


class TestStrictResponseParsing:
    def _run(self, raw_content, seed=None):
        from xssharden.generation.llm import generate_llm_variants

        seed = seed or _seed(sample_id="S001")
        client = _FakeClient(
            lambda messages: {
                "choices": [{"message": {"role": "assistant", "content": raw_content}}]
            }
        )
        return generate_llm_variants([seed], client=client, allow_payload_submission=True)

    def test_malformed_json_is_preserved_as_auditable_error(self):
        result = self._run("not-json{{{")
        assert result.variants == []
        assert len(result.errors) == 1
        assert result.errors[0]["seed_id"] == "S001"
        assert result.errors[0]["raw_content"] == "not-json{{{"
        assert "JSON" in result.errors[0]["reason"]

    def test_top_level_list_is_rejected(self):
        result = self._run('[{"payload": "x"}]')
        assert result.variants == []
        assert len(result.errors) == 1

    def test_missing_variants_key_is_rejected(self):
        result = self._run(json.dumps({"payloads": []}))
        assert result.variants == []
        assert len(result.errors) == 1

    def test_variants_wrong_type_is_rejected(self):
        result = self._run(json.dumps({"variants": {"payload": "x"}}))
        assert result.variants == []
        assert len(result.errors) == 1

    def test_entry_missing_payload_is_rejected(self):
        result = self._run(
            json.dumps({"variants": [{"seed_id": "S001", "mutation_category": "encoding"}]})
        )
        assert result.variants == []
        assert len(result.errors) == 1

    def test_blank_payload_is_rejected(self):
        result = self._run(
            json.dumps(
                {"variants": [{"seed_id": "S001", "payload": "   ", "mutation_category": "encoding"}]}
            )
        )
        assert result.variants == []
        assert len(result.errors) == 1

    def test_wrong_type_payload_is_rejected(self):
        result = self._run(
            json.dumps(
                {"variants": [{"seed_id": "S001", "payload": 123, "mutation_category": "encoding"}]}
            )
        )
        assert result.variants == []
        assert len(result.errors) == 1

    def test_unsupported_mutation_category_is_rejected(self):
        result = self._run(
            json.dumps(
                {"variants": [{"seed_id": "S001", "payload": "<svg>", "mutation_category": "nope"}]}
            )
        )
        assert result.variants == []
        assert len(result.errors) == 1
        assert "nope" in result.errors[0]["reason"]

    def test_missing_mutation_category_is_rejected(self):
        result = self._run(
            json.dumps({"variants": [{"seed_id": "S001", "payload": "<svg>"}]})
        )
        assert result.variants == []
        assert len(result.errors) == 1

    def test_wrong_seed_id_is_rejected(self):
        result = self._run(
            json.dumps(
                {"variants": [{"seed_id": "S999", "payload": "<svg>", "mutation_category": "encoding"}]}
            )
        )
        assert result.variants == []
        assert len(result.errors) == 1
        assert "S999" in result.errors[0]["reason"]

    def test_non_dict_entry_is_rejected(self):
        result = self._run(json.dumps({"variants": ["<svg>"]}))
        assert result.variants == []
        assert len(result.errors) == 1

    def test_missing_choices_envelope_is_rejected(self):
        from xssharden.generation.llm import generate_llm_variants

        client = _FakeClient(lambda messages: {"unexpected": "shape"})
        result = generate_llm_variants(
            [_seed(sample_id="S001")], client=client, allow_payload_submission=True
        )
        assert result.variants == []
        assert len(result.errors) == 1

    def test_valid_variant_preserves_provenance(self):
        from xssharden.generation.llm import generate_llm_variants

        client = _FakeClient(
            lambda messages: _api_response([
                {
                    "seed_id": "S007",
                    "payload": "<ScRiPt>alert(1)</ScRiPt>",
                    "mutation_category": "case_variation",
                    "context_target": "reflected_html",
                }
            ])
        )
        result = generate_llm_variants(
            [_seed(sample_id="S007", source="corpus-a")],
            client=client,
            allow_payload_submission=True,
        )
        assert result.errors == []
        assert len(result.variants) == 1
        variant = result.variants[0]
        assert variant["seed_id"] == "S007"
        assert variant["payload"] == "<ScRiPt>alert(1)</ScRiPt>"
        assert variant["mutation_category"] == "case_variation"
        assert variant["context_target"] == "reflected_html"
        assert variant["source"] == "corpus-a"
        assert variant["generator"] == "llm"
        assert "generator_version" in variant
        assert "variant_id" in variant

    def test_variant_ids_are_deterministic_content_hashes(self):
        from xssharden.generation.llm import generate_llm_variants

        def make_client():
            return _FakeClient(
                lambda messages: _api_response([
                    {
                        "seed_id": "S001",
                        "payload": "<svg onload=prompt(1)>",
                        "mutation_category": "encoding",
                        "context_target": "reflected_html",
                    }
                ])
            )

        first = generate_llm_variants(
            [_seed(sample_id="S001")], client=make_client(), allow_payload_submission=True
        )
        second = generate_llm_variants(
            [_seed(sample_id="S001")], client=make_client(), allow_payload_submission=True
        )
        assert first.variants[0]["variant_id"] == second.variants[0]["variant_id"]

    def test_request_sends_strict_json_contract(self):
        from xssharden.generation.llm import generate_llm_variants

        client = _FakeClient(lambda messages: _api_response([]))
        generate_llm_variants(
            [_seed(sample_id="S001")], client=client, allow_payload_submission=True
        )
        assert len(client.calls) == 1
        dumped = json.dumps(client.calls[0])
        assert "seed_id" in dumped
        assert "mutation_category" in dumped
        assert "variants" in dumped


# ---------------------------------------------------------------------------
# Deduplication (against seeds and within model outputs)
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_variant_matching_seed_payload_is_dropped(self):
        from xssharden.generation.llm import generate_llm_variants

        seed_payload = '<script>alert("xss")</script>'
        client = _FakeClient(
            lambda messages: _api_response([
                {
                    "seed_id": "S001",
                    "payload": seed_payload,
                    "mutation_category": "encoding",
                    "context_target": "reflected_html",
                }
            ])
        )
        result = generate_llm_variants(
            [_seed(sample_id="S001", payload=seed_payload)],
            client=client,
            allow_payload_submission=True,
        )
        assert result.variants == []

    def test_duplicate_payloads_within_one_response_are_deduplicated(self):
        from xssharden.generation.llm import generate_llm_variants

        client = _FakeClient(
            lambda messages: _api_response([
                {
                    "seed_id": "S001",
                    "payload": "<svg onload=prompt(1)>",
                    "mutation_category": "encoding",
                    "context_target": "reflected_html",
                },
                {
                    "seed_id": "S001",
                    "payload": "<svg onload=prompt(1)>",
                    "mutation_category": "case_variation",
                    "context_target": "reflected_html",
                },
            ])
        )
        result = generate_llm_variants(
            [_seed(sample_id="S001")], client=client, allow_payload_submission=True
        )
        assert len(result.variants) == 1

    def test_duplicate_payloads_across_seeds_keep_first(self):
        from xssharden.generation.llm import generate_llm_variants

        shared = "<svg onload=prompt(9)>"
        responses = {
            "S001": [
                {
                    "seed_id": "S001",
                    "payload": shared,
                    "mutation_category": "encoding",
                    "context_target": "reflected_html",
                }
            ],
            "S002": [
                {
                    "seed_id": "S002",
                    "payload": shared,
                    "mutation_category": "encoding",
                    "context_target": "reflected_html",
                }
            ],
        }

        def handler(messages):
            text = json.dumps(messages)
            key = "S002" if "S002" in text else "S001"
            return _api_response(responses[key])

        client = _FakeClient(handler)
        result = generate_llm_variants(
            [
                _seed(sample_id="S001", payload="<script>a</script>"),
                _seed(sample_id="S002", payload="<script>b</script>"),
            ],
            client=client,
            allow_payload_submission=True,
        )
        payloads = [v["payload"] for v in result.variants]
        assert payloads.count(shared) == 1


# ---------------------------------------------------------------------------
# Timeout / transport errors, split boundary, no-execution
# ---------------------------------------------------------------------------


class TestTransportErrorsAndGuards:
    def test_transport_timeout_surfaces_as_typed_timeout_error(self):
        from xssharden.generation.llm import (
            LLMTimeoutError,
            OpenAICompatibleClient,
            LLMConfig,
        )
        import socket

        def exploding_transport(url, headers, body, timeout_s):
            raise socket.timeout("timed out")

        config = LLMConfig(base_url="https://api.example.com/v1", model="m", api_key="k")
        client = OpenAICompatibleClient(config, transport=exploding_transport)
        with pytest.raises(LLMTimeoutError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_transport_connection_failure_surfaces_as_typed_error(self):
        from xssharden.generation.llm import (
            LLMTransportError,
            OpenAICompatibleClient,
            LLMConfig,
        )

        def exploding_transport(url, headers, body, timeout_s):
            raise ConnectionError("refused")

        config = LLMConfig(base_url="https://api.example.com/v1", model="m", api_key="k")
        client = OpenAICompatibleClient(config, transport=exploding_transport)
        with pytest.raises(LLMTransportError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_non_dict_transport_result_is_auditable_error(self):
        from xssharden.generation.llm import generate_llm_variants

        client = _FakeClient(lambda messages: ["not", "a", "dict"])
        result = generate_llm_variants(
            [_seed(sample_id="S001")], client=client, allow_payload_submission=True
        )
        assert result.variants == []
        assert len(result.errors) == 1

    def test_held_out_seeds_are_rejected(self):
        from xssharden.generation.llm import generate_llm_variants

        client = _ok_client()
        with pytest.raises(ValueError, match="not allowed"):
            generate_llm_variants(
                [_seed(sample_id="S001", split="clean-test")],
                client=client,
                allow_payload_submission=True,
            )
        assert client.calls == []

    def test_generation_never_imports_the_browser_validator(self):
        import sys

        from xssharden.generation.llm import generate_llm_variants

        client = _ok_client()
        generate_llm_variants([_seed()], client=client, allow_payload_submission=True)
        assert "xssharden.validation.validator" not in sys.modules
        assert "playwright" not in sys.modules

    def test_generated_payloads_stay_inert_text(self):
        from xssharden.generation.llm import generate_llm_variants

        evil = '<script>alert("xss")</script>'
        client = _FakeClient(
            lambda messages: _api_response([
                {
                    "seed_id": "S001",
                    "payload": evil,
                    "mutation_category": "encoding",
                    "context_target": "reflected_html",
                }
            ])
        )
        result = generate_llm_variants(
            [_seed(sample_id="S001", payload="<svg>")],
            client=client,
            allow_payload_submission=True,
        )
        assert result.variants[0]["payload"] == evil
        assert isinstance(result.variants[0]["payload"], str)

    def test_no_network_without_injected_transport(self):
        import socket
        import urllib.request

        from xssharden.generation.llm import LLMConfig, OpenAICompatibleClient, LLMTimeoutError

        config = LLMConfig(base_url="https://api.example.com/v1", model="m", api_key="k")
        client = OpenAICompatibleClient(config)
        calls = []
        real_urlopen = urllib.request.urlopen

        def spy(*args, **kwargs):
            calls.append(args)
            raise socket.timeout("blocked: network must not be touched in this test")

        urllib.request.urlopen = spy
        try:
            with pytest.raises(LLMTimeoutError):
                client.chat([{"role": "user", "content": "hi"}])
        finally:
            urllib.request.urlopen = real_urlopen
        assert calls, "expected the default transport to attempt urlopen (blocked by spy)"
