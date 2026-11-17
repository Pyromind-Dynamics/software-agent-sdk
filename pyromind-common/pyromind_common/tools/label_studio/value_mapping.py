"""Tolerant resolution between dataset spellings and a config's own values.

Label Studio compares a pre-annotation's value to the control's own list of
values by raw string. When they differ it imports the prediction and draws
nothing, without logging anything: a verdict of ``Defect``, ``defect ``,
``true`` or ``NG`` simply disappears from the review screen. The converter has
to recognise those spellings itself, and it has to do so without a declared
synonym table shadowing the values the config already accepts.

Resolution runs in one fixed order, so a later step can only add recognition --
it never replaces a value the config already accepts, and never adopts a mapping
to a value the config cannot render:

1. the control's own values, matched case/width/separator-insensitively;
2. the binding's synonym table, then the built-in verdict vocabulary, each
   re-checked against the control's values so a mapping to a value the config
   does not list is not adopted;
3. a conservative fuzzy match against the control's own values.

A value that survives all three is reported unmatched, so the caller can warn
instead of writing it where it cannot render.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal


# Verdict/boolean spellings our pipelines and the inspection systems they read
# have emitted, keyed by canonical verdict. Applied to a control only after its
# own values have been checked, and only kept when the control lists the target,
# so this widens recognition without ever overriding the config.
VERDICT_SYNONYMS: dict[str, str] = {
    "defect": "defect",
    "true": "defect",
    "yes": "defect",
    "positive": "defect",
    "bad": "defect",
    "ng": "defect",
    "fault": "defect",
    "faulty": "defect",
    "fail": "defect",
    "failed": "defect",
    "abnormal": "defect",
    "error": "defect",
    "anomaly": "defect",
    "1": "defect",
    "ok": "ok",
    "good": "ok",
    "fine": "ok",
    "okay": "ok",
    "pass": "ok",
    "passed": "ok",
    "no": "ok",
    "negative": "ok",
    "normal": "ok",
    "0": "ok",
    "false_positive": "ok",
    "false": "ok",
}

# Case, width and separator differences that are not meaningful in a label or
# choice. Folding them lets "Defect", "defect " and "短 路" match a config value
# that was written without them.
_SEPARATORS = re.compile(r"[\s\-_/\\.、,，;；:：·・|()（）\[\]{}【】<>《》\"'“”‘’]+")

# A near-miss value is only adopted when one config value is clearly closest:
# difflib ratio at or above this, with no tie. Below it the spelling is treated
# as genuinely different rather than guessed at.
_FUZZY_THRESHOLD = 0.8


def normalize_value_key(value: str) -> str:
    """Return the case/width/separator-insensitive form of a control value."""
    folded = unicodedata.normalize("NFKC", value).strip().lower()
    return _SEPARATORS.sub("", folded)


def _normalized_table(table: Mapping[str, str] | None) -> dict[str, str]:
    if not table:
        return {}
    return {normalize_value_key(key): value for key, value in table.items()}


@dataclass(frozen=True)
class ValueMatch:
    """The value to write for one datum, and how it was arrived at.

    ``strategy`` is ``"config"`` for a value the config already lists,
    ``"table"`` for a synonym table hit, ``"fuzzy"`` for a near-miss guess, and
    ``"none"`` when nothing matched -- at which point ``value`` is the raw text
    and the caller decides between dropping it and recording it.
    """

    value: str
    strategy: Literal["config", "table", "fuzzy", "none"]

    @property
    def matched(self) -> bool:
        """Whether the config can render this value."""
        return self.strategy != "none"


class ControlValueIndex:
    """The values one control can render, with a tolerant lookup."""

    def __init__(self, values: Collection[str] | None = None) -> None:
        self._by_key: dict[str, str] = {}
        for value in values or ():
            normalized = normalize_value_key(value)
            if normalized:
                self._by_key.setdefault(normalized, value)

    def __bool__(self) -> bool:
        return bool(self._by_key)

    def exact(self, text: str) -> str | None:
        """The config's own spelling of ``text``, ignoring case and separators."""
        return self._by_key.get(normalize_value_key(text))

    def closest(self, text: str) -> str | None:
        """The nearest config value, when exactly one is clearly closest."""
        key = normalize_value_key(text)
        if not key:
            return None
        exact = self._by_key.get(key)
        if exact is not None:
            return exact
        best_ratio = 0.0
        best_value: str | None = None
        ambiguous = False
        for candidate_key, value in self._by_key.items():
            ratio = difflib.SequenceMatcher(None, key, candidate_key).ratio()
            if ratio < _FUZZY_THRESHOLD:
                continue
            if ratio > best_ratio:
                best_ratio = ratio
                best_value = value
                ambiguous = False
            elif best_value is not None:
                ambiguous = True
        return None if ambiguous or best_value is None else best_value


def resolve_control_value(
    text: str,
    *,
    config_values: ControlValueIndex | Collection[str] | None = None,
    synonyms: Mapping[str, str] | None = None,
    builtin: Mapping[str, str] | None = None,
) -> ValueMatch:
    """Resolve ``text`` to the spelling a control can render.

    Returns the resolved value together with how it was resolved. An unmatched
    value comes back verbatim with ``strategy="none"`` so the caller can decide
    between dropping it and recording it as an unmapped spelling.
    """
    index = (
        config_values
        if isinstance(config_values, ControlValueIndex)
        else ControlValueIndex(config_values)
    )
    if index:
        exact = index.exact(text)
        if exact is not None:
            return ValueMatch(exact, "config")

    for table in (_normalized_table(synonyms), _normalized_table(builtin)):
        mapped = table.get(normalize_value_key(text))
        if mapped is None:
            continue
        if not index:
            # No declared values to check against, so the table is the only
            # statement of intent there is.
            return ValueMatch(mapped, "table")
        known = index.exact(mapped)
        if known is not None:
            return ValueMatch(known, "table")

    if index:
        fuzzy = index.closest(text)
        if fuzzy is not None:
            return ValueMatch(fuzzy, "fuzzy")

    return ValueMatch(text, "none")
