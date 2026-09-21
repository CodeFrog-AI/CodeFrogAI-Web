"""Keep secrets out of AI context: withhold sensitive files and redact secret-shaped text.

Redaction is heuristic. It targets common credential formats and deliberately leaves
ordinary code alone (`token_count = 5`, `password = get_password()`).
"""

import re
from collections.abc import Callable
from pathlib import PurePosixPath

REDACTION = "[REDACTED]"

_SENSITIVE_NAMES = frozenset({".npmrc", ".pypirc"})
_SENSITIVE_NAME_PREFIXES = (".env", "id_rsa", "service-account")
_SENSITIVE_NAME_SUFFIXES = (".pem", ".key")
_SENSITIVE_PATH_WORDS = ("secret", "credential")


def is_sensitive_path(path: str) -> bool:
    """True for files that must never be returned: env files, keys, and credential stores.

    The `secret`/`credential` words are checked in every path segment, so files inside
    a `secrets/` directory are withheld too.
    """

    parts = [part.lower() for part in PurePosixPath(path).parts]
    if not parts:
        return False
    name = parts[-1]
    if (
        name in _SENSITIVE_NAMES
        or name.startswith(_SENSITIVE_NAME_PREFIXES)
        or name.endswith(_SENSITIVE_NAME_SUFFIXES)
    ):
        return True
    return any(word in part for part in parts for word in _SENSITIVE_PATH_WORDS)


Rule = Callable[[str], tuple[str, int]]

_NAME = (
    r"(?:password|passwd|[_.-]pass|secret|secret[_-]?(?:access[_-]?)?key|private[_-]?key"
    r"|token|api[_-]?key|apikey)"
)
# The sensitive word must END the identifier: `access_token` yes, `token_count`/`tokens` no.
# The prefix is bounded so crafted dotted/hyphenated runs cannot cause quadratic backtracking.
_ASSIGNED_NAME = rf"[\w.-]{{0,64}}{_NAME}[\"']?"
_ANNOTATION = r"(?:\s*:\s*[A-Za-z_][\w\[\], |.]*)?"

_QUOTED_ASSIGNMENT = re.compile(
    rf"(?i)({_ASSIGNED_NAME}{_ANNOTATION}\s*[:=]\s*)([\"'])([^\"'\n]{{6,}})(\2)"
)
_UNQUOTED_ASSIGNMENT = re.compile(
    rf"(?i)({_ASSIGNED_NAME}\s*[:=]\s*)([A-Za-z0-9_+/=-]{{8,}})(?![\w(.])"
)
_PLAIN_VALUES = frozenset({"true", "false", "none", "null"})


def _pattern_rule(pattern: str, replacement: str | Callable[[re.Match[str]], str], flags: int = 0) -> Rule:
    compiled = re.compile(pattern, flags)
    return lambda text: compiled.subn(replacement, text)


def _line_preserving(match: re.Match[str]) -> str:
    # Keep the line count so reported line numbers stay valid.
    return REDACTION + "\n" * match.group(0).count("\n")


def _quoted_assignment(text: str) -> tuple[str, int]:
    return _QUOTED_ASSIGNMENT.subn(lambda m: f"{m.group(1)}{m.group(2)}{REDACTION}{m.group(4)}", text)


def _unquoted_assignment(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group(2)
        looks_secret = (
            any(character.isdigit() for character in value)
            and any(character.isalpha() for character in value)
            and value.lower() not in _PLAIN_VALUES
        )
        if not looks_secret:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{REDACTION}"

    return _UNQUOTED_ASSIGNMENT.sub(replace, text), count


_RULES: tuple[Rule, ...] = (
    # PEM private keys, including a key cut off by a snippet boundary on either side.
    _pattern_rule(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
        _line_preserving,
        re.S,
    ),
    _pattern_rule(
        r"(?:^[ \t]*[A-Za-z0-9+/=]{16,}[ \t]*\n)*[ \t]*-----END [A-Z0-9 ]*PRIVATE KEY-----",
        _line_preserving,
        re.M,
    ),
    _pattern_rule(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", REDACTION),
    _pattern_rule(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", REDACTION),
    _pattern_rule(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", REDACTION),
    _pattern_rule(r"\bsk-[A-Za-z0-9_-]{16,}", REDACTION),
    _pattern_rule(r"\bxox[abposr]-[A-Za-z0-9-]{10,}", REDACTION),
    _pattern_rule(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}", REDACTION),
    _pattern_rule(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]{16,}", rf"\1{REDACTION}", re.I),
    # The username may be empty: `redis://:password@host`.
    _pattern_rule(r"(\b[a-z][a-z0-9+.-]{0,31}://[^\s/:@]*:)[^\s/@]{3,}(@)", rf"\1{REDACTION}\2", re.I),
    _quoted_assignment,
    _unquoted_assignment,
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Replace secret-shaped values with a placeholder; returns (text, replacements)."""

    total = 0
    for rule in _RULES:
        text, count = rule(text)
        total += count
    return text, total
