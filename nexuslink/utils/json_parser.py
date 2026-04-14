"""Robust JSON extraction from LLM response text.

Claude sometimes wraps JSON in markdown fences or prepends prose — this module
handles all common response shapes with a clear error when nothing parses.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | list:
    """Extract and parse JSON from *text*, tolerating LLM response wrappers.

    Attempts (in order):
    1. Direct ``json.loads`` of the full stripped text.
    2. Content inside a ``\\`\\`\\`json`` or bare ``\\`\\`\\`` fence.
    3. First ``[...]`` or ``{...}`` block found anywhere in the text.

    Raises
    ------
    ValueError
        If no valid JSON is found after all attempts.
    """
    stripped = text.strip()

    # 1. Direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block  (```json ... ``` or ``` ... ```)
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced [ ] or { } in the text
    bracket = re.search(r"(\[[\s\S]+\]|\{[\s\S]+\})", stripped)
    if bracket:
        try:
            return json.loads(bracket.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"No valid JSON found in LLM response. "
        f"First 300 chars: {stripped[:300]!r}"
    )
