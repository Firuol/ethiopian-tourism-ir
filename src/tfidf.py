"""TF-IDF retrieval with cosine similarity.

A term weighs more when it is frequent in a document and rare in the
collection. Document and query vectors are length-normalized, so a longer
page is not preferred only because it contains more words.
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


def _tf_weight(frequency: int) -> float:
    if frequency <= 0:
        return 0.0
    return 1.0 + math.log(frequency)


class TfidfRetriever:
    def __init__(self, index: InvertedIndex) -> None:
        self.index = index
        self.idf: dict[str, float] = {}
        for term, doc_freq in index.doc_freq.items():
            self.idf[term] = math.log(index.num_docs / doc_freq) if doc_freq else 0.0
        norm_sq = {doc_id: 0.0 for doc_id in index.documents}
        for term, postings in index.postings.items():
            idf = self.idf[term]
            if idf == 0.0:
                continue
            for doc_id, frequency in postings:
                weight = _tf_weight(frequency) * idf
                norm_sq[doc_id] += weight * weight
        self.doc_norm = {
            doc_id: math.sqrt(total) for doc_id, total in norm_sq.items()
        }

    def search(self, query: str, k: int = 10) -> list[dict]:
        terms = preprocess(query)
        if not terms or k <= 0:
            return []
        query_weights: dict[str, float] = {}
        query_norm_sq = 0.0
        for term, frequency in Counter(terms).items():
            idf = self.idf.get(term, 0.0)
            if idf == 0.0:
                continue
            weight = _tf_weight(frequency) * idf
            query_weights[term] = weight
            query_norm_sq += weight * weight
        if query_norm_sq == 0.0:
            return []
        query_norm = math.sqrt(query_norm_sq)

        dots: dict[str, float] = {}
        for term, query_weight in query_weights.items():
            idf = self.idf[term]
            for doc_id, frequency in self.index.term_postings(term):
                doc_weight = _tf_weight(frequency) * idf
                dots[doc_id] = dots.get(doc_id, 0.0) + query_weight * doc_weight

        ranked: list[tuple[float, str]] = []
        for doc_id, dot in dots.items():
            denom = query_norm * self.doc_norm.get(doc_id, 0.0)
            if denom == 0.0:
                continue
            ranked.append((dot / denom, doc_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        hits = []
        for score, doc_id in ranked[:k]:
            hit = dict(self.index.documents[doc_id])
            hit["score"] = score
            hits.append(hit)
        return hits


if __name__ == "__main__":
    retriever = TfidfRetriever(InvertedIndex.build())
    queries = [
        "rock hewn churches of lalibela",
        "simien mountains trek",
        "ethiopian coffee ceremony",
    ]
    for query in queries:
        print(f"\n{query}")
        for rank, hit in enumerate(retriever.search(query, k=5), start=1):
            print(f"  {rank}. {hit['score']:.3f}  {hit['title']}")
