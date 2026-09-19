"""Turn a natural-language question into a few deterministic exact-search terms."""

import re

MAX_TERMS = 5
MAX_TERM_LENGTH = 100
MIN_CODE_TERM_LENGTH = 4

STOPWORDS = frozenset(
    {
        "about", "above", "after", "again", "also", "another", "because", "been", "before",
        "being", "between", "both", "cannot", "class", "classes", "code", "codebase", "could",
        "defined", "does", "done", "each", "either", "else", "every", "exist", "exists",
        "explain", "file", "files", "find", "from", "function", "functions", "give", "handle",
        "handled", "handles", "handling", "happen", "happens", "have", "having", "here",
        "implementation", "implemented", "into", "just", "like", "list", "located", "made",
        "make", "many", "method", "methods", "might", "more", "most", "much", "must", "need",
        "needs", "only", "other", "over", "please", "repo", "repository", "same", "show",
        "should", "some", "such", "tell", "than", "that", "their", "them", "then", "there",
        "these", "they", "this", "those", "through", "under", "used", "uses", "using", "very",
        "want", "were", "what", "when", "where", "which", "while", "will", "with", "within",
        "without", "work", "works", "would", "your",
    }
)

_QUOTED = re.compile(r"`([^`\n]+)`|\"([^\"\n]+)\"|(?<!\w)'([^'\n]+)'(?!\w)")
_EDGE_PUNCTUATION = ".,;:!?()[]{}<>\"'"
_CAMEL_CASE = re.compile(r"[a-z0-9][A-Z]")
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_PATH = re.compile(r"^[\w.\-]+(?:/[\w.\-]+)+/?$")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")


def _is_code_like(token: str) -> bool:
    has_letter = any(character.isalpha() for character in token)
    return has_letter and (
        "_" in token
        or bool(_CAMEL_CASE.search(token))
        or bool(_DOTTED.match(token))
        or bool(_PATH.match(token))
    )


def extract_terms(question: str) -> list[str]:
    """Return up to MAX_TERMS search terms, most specific first.

    Order: quoted or backticked spans, then code-like tokens (snake_case, camelCase,
    dotted names, paths), then plain words of 4+ characters that are not stopwords.
    Terms are de-duplicated case-insensitively; the first spelling wins.
    """

    quoted = [
        span
        for match in _QUOTED.finditer(question)
        if 2 <= len(span := next(group for group in match.groups() if group is not None).strip()) <= MAX_TERM_LENGTH
    ]

    code_terms: list[str] = []
    words: list[str] = []
    for raw in _QUOTED.sub(" ", question).split():
        token = raw.strip(_EDGE_PUNCTUATION)
        if len(token) < MIN_CODE_TERM_LENGTH:
            continue
        if _is_code_like(token):
            if len(token) <= MAX_TERM_LENGTH:
                code_terms.append(token)
            continue
        words.extend(word for word in _WORD.findall(token) if word.lower() not in STOPWORDS)

    terms: list[str] = []
    seen: set[str] = set()
    for candidate in (*quoted, *code_terms, *words):
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(candidate)
        if len(terms) == MAX_TERMS:
            break
    return terms
