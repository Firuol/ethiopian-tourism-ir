"""Boolean retrieval over the inverted index.

A query with no operators is treated as AND: every remaining term must occur.
AND, OR, and NOT are accepted, and parentheses set the grouping.
NOT binds tightest, then AND, then OR.
"""

from __future__ import annotations

import re

try:
    from .inverted_index import InvertedIndex
    from .preprocessing import preprocess
except ImportError:
    from inverted_index import InvertedIndex
    from preprocessing import preprocess

OPERATORS = {"AND", "OR", "NOT"}
TOKEN_PATTERN = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[A-Za-z']+", re.IGNORECASE)


def _intersect(left: list[str], right: list[str]) -> list[str]:
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


def _union(left: list[str], right: list[str]) -> list[str]:
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def _difference(left: list[str], right: list[str]) -> list[str]:
    right_ids = set(right)
    return [doc_id for doc_id in left if doc_id not in right_ids]


class BooleanRetriever:
    def __init__(self, index: InvertedIndex) -> None:
        self.index = index
        self._all_ids = index.document_ids()

    def search(self, query: str) -> list[dict]:
        doc_ids = self._evaluate(query)
        return [self.index.documents[doc_id] for doc_id in doc_ids]

    def _postings(self, term: str) -> list[str]:
        return [doc_id for doc_id, _frequency in self.index.term_postings(term)]

    def _tokens(self, query: str) -> list[str]:
        raw = TOKEN_PATTERN.findall(query)
        has_operator = any(token.upper() in OPERATORS or token in "()" for token in raw)
        if not has_operator:
            return ["AND", *preprocess(query)] if preprocess(query) else []
        tokens = []
        for token in raw:
            upper = token.upper()
            if upper in OPERATORS or token in "()":
                tokens.append(upper if upper in OPERATORS else token)
            else:
                terms = preprocess(token)
                tokens.extend(terms)
        return tokens

    def _evaluate(self, query: str) -> list[str]:
        tokens = self._tokens(query)
        if not tokens:
            return []
        # A leading AND is only a marker for an implicit conjunction.
        if tokens[0] == "AND":
            tokens = tokens[1:]
        if not tokens:
            return []
        self._tokens_left = tokens
        self._position = 0
        result = self._parse_or()
        return result

    def _peek(self) -> str | None:
        if self._position >= len(self._tokens_left):
            return None
        return self._tokens_left[self._position]

    def _take(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of query")
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected}, found {token}")
        self._position += 1
        return token

    def _parse_or(self) -> list[str]:
        result = self._parse_and()
        while self._peek() == "OR":
            self._take("OR")
            result = _union(result, self._parse_and())
        return result

    def _parse_and(self) -> list[str]:
        result = self._parse_not()
        while self._peek() not in (None, "OR", ")"):
            if self._peek() == "AND":
                self._take("AND")
            result = _intersect(result, self._parse_not())
        return result

    def _parse_not(self) -> list[str]:
        if self._peek() == "NOT":
            self._take("NOT")
            return _difference(self._all_ids, self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> list[str]:
        token = self._peek()
        if token == "(":
            self._take("(")
            result = self._parse_or()
            self._take(")")
            return result
        if token is None or token in OPERATORS:
            raise ValueError(f"Unexpected token in query: {token}")
        self._take()
        return self._postings(token)


def _titles(hits: list[dict], limit: int = 5) -> str:
    return "; ".join(hit["title"] for hit in hits[:limit]) or "(none)"


if __name__ == "__main__":
    retriever = BooleanRetriever(InvertedIndex.build())
    queries = [
        "lalibela churches",
        "simien AND park",
        "injera OR kitfo",
        "park NOT simien",
    ]
    for query in queries:
        hits = retriever.search(query)
        print(f"{query}: {len(hits)} hits")
        print(" ", _titles(hits))
