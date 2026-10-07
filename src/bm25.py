"""BM25 retrieval.

BM25 scores a document from the query terms it contains. Repeated terms help,
but with diminishing effect, and the score is adjusted for document length.
Default parameters are k1 = 1.5 and b = 0.75.
"""

from __future__ import annotations

import math
from collections import Counter

try:
    from .inverted_index import InvertedIndex
    from .preprocessing import preprocess
except ImportError:
    from inverted_index import InvertedIndex
    from preprocessing import preprocess


class BM25Retriever:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        self.idf: dict[str, float] = {}
        num_docs = index.num_docs
        for term, doc_freq in index.doc_freq.items():
            self.idf[term] = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, k: int = 10) -> list[dict]:
        terms = preprocess(query)
        if not terms or k <= 0:
            return []
        scores: dict[str, float] = {}
        avg_len = self.index.avg_doc_len or 1.0
        for term, query_frequency in Counter(terms).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, frequency in self.index.term_postings(term):
                doc_len = self.index.doc_len[doc_id]
                denom = frequency + self.k1 * (1.0 - self.b + self.b * doc_len / avg_len)
                weight = idf * (frequency * (self.k1 + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * query_frequency

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        hits = []
        for doc_id, score in ranked[:k]:
            hit = dict(self.index.documents[doc_id])
            hit["score"] = score
            hits.append(hit)
        return hits


if __name__ == "__main__":
    retriever = BM25Retriever(InvertedIndex.build())
    queries = [
        "rock hewn churches of lalibela",
        "simien mountains trek",
        "ethiopian coffee ceremony",
    ]
    for query in queries:
        print(f"\n{query}")
        for rank, hit in enumerate(retriever.search(query, k=5), start=1):
            print(f"  {rank}. {hit['score']:.3f}  {hit['title']}")
