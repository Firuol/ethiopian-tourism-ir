"""Query expansion for tourism search.

Two steps are applied before the second retrieval:

1. Spelling and name variants that refer to the same place or custom, such as
   Axum and Aksum, or Simien and Semien.
2. Pseudo-relevance feedback: distinctive terms from the first few hits are
   added, then the query is run again.
"""

from __future__ import annotations

from collections import Counter

try:
    from .preprocessing import preprocess
except ImportError:
    from preprocessing import preprocess

# Surface forms. They are stemmed when the synonym map is built.
SYNONYM_GROUPS = [
    ["axum", "aksum"],
    ["simien", "semien", "simen"],
    ["mekelle", "mekele"],
    ["hawassa", "awasa", "awassa"],
    ["harar", "harer"],
    ["gondar", "gonder"],
    ["injera", "enjera"],
    ["timkat", "timket"],
    ["meskel", "maskal"],
    ["irreecha", "ireecha", "irrecha"],
    ["tej", "tedj"],
]

# If every term of one phrase is in the query, add the other phrase.
PHRASE_PAIRS = [
    (["blue", "nile", "falls"], ["tis", "abay"]),
    (["tis", "abay"], ["blue", "nile", "falls"]),
    (["tis", "isat"], ["blue", "nile", "falls"]),
]


def _stem(word: str) -> str | None:
    terms = preprocess(word)
    if len(terms) == 1:
        return terms[0]
    return None


def _build_synonyms() -> dict[str, list[str]]:
    synonyms: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        stems = []
        for word in group:
            stem = _stem(word)
            if stem:
                stems.append((stem, word))
        for stem, _word in stems:
            synonyms.setdefault(stem, set())
            for other_stem, other_word in stems:
                if other_stem != stem:
                    synonyms[stem].add(other_word)
    return {stem: sorted(words) for stem, words in synonyms.items()}


SYNONYMS = _build_synonyms()


def synonym_terms(query: str) -> list[str]:
    """Extra words for known name variants present in the query."""
    query_terms = set(preprocess(query))
    extra: list[str] = []
    seen = set(query_terms)
    for term in list(query_terms):
        for word in SYNONYMS.get(term, []):
            stem = _stem(word)
            if stem and stem not in seen:
                extra.append(word)
                seen.add(stem)
    for source, target in PHRASE_PAIRS:
        source_stems = [stem for word in source if (stem := _stem(word))]
        if source_stems and all(stem in query_terms for stem in source_stems):
            for word in target:
                stem = _stem(word)
                if stem and stem not in seen:
                    extra.append(word)
                    seen.add(stem)
    return extra


def feedback_terms(retriever, query: str, top_docs: int = 5, n_terms: int = 3) -> list[str]:
    """Terms that stand out in the first retrieved documents."""
    hits = retriever.search(query, k=top_docs)
    if not hits:
        return []
    index = retriever.index
    query_terms = set(preprocess(query))
    counts: Counter[str] = Counter()
    for hit in hits:
        counts.update(preprocess(f"{hit['title']}\n{hit['text']}"))
    scored: list[tuple[float, str]] = []
    for term, frequency in counts.items():
        if term in query_terms or len(term) < 3:
            continue
        doc_freq = index.doc_freq.get(term, 0)
        if doc_freq < 2 or doc_freq > index.num_docs * 0.25:
            continue
        idf = getattr(retriever, "idf", {}).get(term, 1.0)
        scored.append((frequency * idf, term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [term for _score, term in scored[:n_terms]]


def expand_query(
    query: str,
    retriever=None,
    use_synonyms: bool = True,
    use_feedback: bool = True,
    top_docs: int = 5,
    n_terms: int = 3,
) -> str:
    """Return the query with added variant names and feedback terms."""
    parts = [query.strip()]
    if use_synonyms:
        parts.extend(synonym_terms(query))
    if use_feedback and retriever is not None:
        seed = " ".join(parts)
        for term in feedback_terms(retriever, seed, top_docs=top_docs, n_terms=n_terms):
            if term not in preprocess(seed):
                parts.append(term)
    return " ".join(part for part in parts if part)


class ExpandedRetriever:
    """Runs a retriever on the expanded query."""

    def __init__(
        self,
        retriever,
        use_synonyms: bool = True,
        use_feedback: bool = True,
        top_docs: int = 5,
        n_terms: int = 3,
    ) -> None:
        self.retriever = retriever
        self.use_synonyms = use_synonyms
        self.use_feedback = use_feedback
        self.top_docs = top_docs
        self.n_terms = n_terms

    def expand(self, query: str) -> str:
        return expand_query(
            query,
            retriever=self.retriever,
            use_synonyms=self.use_synonyms,
            use_feedback=self.use_feedback,
            top_docs=self.top_docs,
            n_terms=self.n_terms,
        )

    def search(self, query: str, k: int = 10) -> list[dict]:
        return self.retriever.search(self.expand(query), k=k)


if __name__ == "__main__":
    try:
        from .bm25 import BM25Retriever
        from .inverted_index import InvertedIndex
    except ImportError:
        from bm25 import BM25Retriever
        from inverted_index import InvertedIndex

    base = BM25Retriever(InvertedIndex.build())
    expanded = ExpandedRetriever(base)
    for query in ("axum obelisk", "simien national park", "blue nile falls"):
        print(f"\n{query}")
        print(" expanded:", expanded.expand(query))
        print(" before:", "; ".join(hit["title"] for hit in base.search(query, k=3)))
        print(" after: ", "; ".join(hit["title"] for hit in expanded.search(query, k=3)))
