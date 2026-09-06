#!/usr/bin/env python3
"""Quick test of LLM variant generation via OpenRouter.

Usage:
    export XSSHARDEN_LLM_API_KEY="sk-or-your-key-here"
    python test_llm.py

Generate 10 variants from 3 seed payloads to verify the integration works.
"""

import json
import os
import sys

# Check API key
api_key = os.environ.get("XSSHARDEN_LLM_API_KEY", "")
if not api_key:
    print("ERROR: Set XSSHARDEN_LLM_API_KEY first:")
    print('  export XSSHARDEN_LLM_API_KEY="sk-or-your-key-here"')
    sys.exit(1)

# Configure
os.environ["XSSHARDEN_LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["XSSHARDEN_LLM_MODEL"] = "nvidia/nemotron-3.5-lightning:free"

from xssharden.generation.llm import (
    LLMConfig,
    OpenAICompatibleClient,
    build_variant_messages,
    generate_llm_variants,
)

config = LLMConfig.from_env()
client = OpenAICompatibleClient(config)

# Test with 3 seed payloads
seeds = [
    {"sample_id": "test-1", "payload": '<script>alert("xss")</script>',
     "label": 1, "source": "test", "attack_category": "xss"},
    {"sample_id": "test-2", "payload": '<img src=x onerror=prompt(1)>',
     "label": 1, "source": "test", "attack_category": "xss"},
    {"sample_id": "test-3", "payload": '<svg onload=confirm(1)>',
     "label": 1, "source": "test", "attack_category": "xss"},
]

print(f"Model: {config.model}")
print(f"Endpoint: {config.base_url}")
print(f"Generating 3 variants per seed ({len(seeds)} seeds)...")
print()

# Single request test first
print("--- Single request test ---")
messages = build_variant_messages(seeds[0], ["encoding", "whitespace_comment",
    "tag_event_substitution", "case_variation"], variants_per_seed=2)
print(f"Prompt: {messages[1]['content'][:200]}...")
print()

import time

def chat_with_retry(client, messages, max_retries=5, base_delay=10):
    """Chat with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            return client.chat(messages)
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"  Rate limited, waiting {delay}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")

try:
    response = chat_with_retry(client, messages)
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Raw response:\n{content[:500]}")
    print()

    # Parse
    from xssharden.generation.llm import parse_model_content
    parsed = parse_model_content(content, seed_id="test-1")
    for item in parsed:
        print(f"  {item['mutation_category']}: {item['payload']}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)

print()

# Full generation test
print("--- Full generation test ---")
try:
    # Patch the client's chat method to include retries
    original_chat = client.chat
    def chat_with_retry_inner(messages):
        return chat_with_retry(client, messages)
    client.chat = chat_with_retry_inner

    result = generate_llm_variants(
        seeds,
        client=client,
        allow_payload_submission=True,
        variants_per_seed=2,
    )
    print(f"Generated: {len(result.variants)} variants, {len(result.errors)} errors")
    for v in result.variants:
        print(f"  [{v['mutation_category']}] {v['payload'][:80]}")
    for e in result.errors:
        print(f"  ERROR: {e.get('reason', 'unknown')}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)

print()
print("SUCCESS — LLM generation works!")
