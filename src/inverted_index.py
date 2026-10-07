"""Inverted index built from preprocessed document terms.

Each term maps to a postings list of (doc_id, term frequency), sorted by
document id. Document length is stored as well, for later ranking.
"""

from __future__ import annotations

from collections import Counter

try:
    from .preprocessing import load_documents
except ImportError:
    from preprocessing import load_documents


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, list[tuple[str, int]]] = {}
        self.doc_freq: dict[str, int] = {}
        self.doc_len: dict[str, int] = {}
        self.documents: dict[str, dict] = {}
        self.num_docs = 0
        self.avg_doc_len = 0.0

    def add(self, document: dict) -> None:
        doc_id = document["doc_id"]
        terms = document["terms"]
        counts = Counter(terms)
        self.doc_len[doc_id] = len(terms)
        self.documents[doc_id] = {
            "doc_id": doc_id,
            "title": document["title"],
            "url": document["url"],
            "source": document["source"],
            "topic": document["topic"],
            "text": document["text"],
        }
        for term, frequency in counts.items():
            self.postings.setdefault(term, []).append((doc_id, frequency))

    def finalize(self) -> None:
        for term, postings in self.postings.items():
            postings.sort(key=lambda posting: posting[0])
            self.doc_freq[term] = len(postings)
        self.num_docs = len(self.documents)
        if self.num_docs:
            self.avg_doc_len = sum(self.doc_len.values()) / self.num_docs

    @classmethod
    def build(cls, documents: list[dict] | None = None) -> "InvertedIndex":
        index = cls()
        for document in documents if documents is not None else load_documents():
            index.add(document)
        index.finalize()
        return index

    def document_ids(self) -> list[str]:
        return sorted(self.documents)

    def term_postings(self, term: str) -> list[tuple[str, int]]:
        return self.postings.get(term, [])


if __name__ == "__main__":
    index = InvertedIndex.build()
    print(f"Documents: {index.num_docs}")
    print(f"Vocabulary: {len(index.postings)}")
    print(f"Average document length: {index.avg_doc_len:.1f} terms")
    for term in ("lalibela", "church", "park"):
        print(f"  {term}: {index.doc_freq.get(term, 0)} documents")
