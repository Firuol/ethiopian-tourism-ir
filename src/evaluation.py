"""Compare retrieval models with Precision@k, Recall, MAP, and nDCG.

Relevance grades in data/qrels.csv are 0, 1, and 2. A document is relevant for
precision, recall, and MAP when its grade is at least 1. nDCG uses the full
grade. Unjudged documents are treated as not relevant.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

try:
    from .bm25 import BM25Retriever
    from .boolean import BooleanRetriever
    from .inverted_index import InvertedIndex
    from .query_expansion import ExpandedRetriever
    from .tfidf import TfidfRetriever
except ImportError:
    from bm25 import BM25Retriever
    from boolean import BooleanRetriever
    from inverted_index import InvertedIndex
    from query_expansion import ExpandedRetriever
    from tfidf import TfidfRetriever

ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data" / "queries.json"
QRELS_PATH = ROOT / "data" / "qrels.csv"

K = 10
RANK_DEPTH = 100


def load_queries(path: Path | None = None) -> list[dict]:
    return json.loads((path or QUERIES_PATH).read_text(encoding="utf-8"))


def load_qrels(path: Path | None = None) -> dict[str, dict[str, int]]:
    judgments: dict[str, dict[str, int]] = {}
    with (path or QRELS_PATH).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            judgments.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return judgments


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(relevant)


def average_precision(ranked_ids: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hit_count = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hit_count += 1
            total += hit_count / rank
    return total / len(relevant)


def dcg_at_k(grades: list[int], k: int) -> float:
    score = 0.0
    for rank, grade in enumerate(grades[:k], start=1):
        score += (2**grade - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    gained = [grades.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(grades.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(gained, k) / ideal_dcg


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = K) -> dict:
    """`run` maps query_id to document ids in rank order."""
    per_query = []
    for query_id, ranked_ids in run.items():
        grades = qrels.get(query_id, {})
        relevant = {doc_id for doc_id, grade in grades.items() if grade >= 1}
        per_query.append(
            {
                "query_id": query_id,
                "precision": precision_at_k(ranked_ids, relevant, k),
                "recall": recall_at_k(ranked_ids, relevant, k),
                "average_precision": average_precision(ranked_ids, relevant),
                "ndcg": ndcg_at_k(ranked_ids, grades, k),
                "top_relevant": ranked_ids[0] in relevant if ranked_ids else False,
            }
        )
    count = len(per_query) or 1
    return {
        "per_query": per_query,
        "precision": sum(row["precision"] for row in per_query) / count,
        "recall": sum(row["recall"] for row in per_query) / count,
        "map": sum(row["average_precision"] for row in per_query) / count,
        "ndcg": sum(row["ndcg"] for row in per_query) / count,
    }


def build_models(index: InvertedIndex) -> dict:
    boolean = BooleanRetriever(index)
    tfidf = TfidfRetriever(index)
    bm25 = BM25Retriever(index)
    return {
        "Boolean": boolean,
        "TF-IDF": tfidf,
        "BM25": bm25,
        "BM25+expansion": ExpandedRetriever(bm25),
    }


def run_model(retriever, queries: list[dict], depth: int = RANK_DEPTH) -> dict[str, list[str]]:
    run = {}
    for query in queries:
        try:
            hits = retriever.search(query["text"], k=depth)
        except TypeError:
            hits = retriever.search(query["text"])[:depth]
        run[query["query_id"]] = [hit["doc_id"] for hit in hits]
    return run


def failed_queries(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], queries: list[dict]) -> list[dict]:
    """Queries whose top document is not relevant, or that miss every highly relevant document."""
    text = {query["query_id"]: query["text"] for query in queries}
    failures = []
    for query_id, ranked_ids in run.items():
        grades = qrels.get(query_id, {})
        highly = {doc_id for doc_id, grade in grades.items() if grade >= 2}
        top = ranked_ids[0] if ranked_ids else None
        top_grade = grades.get(top, 0) if top else 0
        highly_in_top = [doc_id for doc_id in ranked_ids[:K] if doc_id in highly]
        if top_grade == 0 or not highly_in_top:
            failures.append(
                {
                    "query_id": query_id,
                    "text": text.get(query_id, ""),
                    "top_doc": top,
                    "top_grade": top_grade,
                    "highly_relevant_in_top_k": len(highly_in_top),
                }
            )
    return failures


def compare(index: InvertedIndex | None = None) -> dict[str, dict]:
    index = index or InvertedIndex.build()
    queries = load_queries()
    qrels = load_qrels()
    results = {}
    runs = {}
    for name, retriever in build_models(index).items():
        run = run_model(retriever, queries)
        runs[name] = run
        results[name] = evaluate_run(run, qrels)
        results[name]["failures"] = failed_queries(run, qrels, queries)
    results["_runs"] = runs
    results["_queries"] = queries
    results["_qrels"] = qrels
    results["_index"] = index
    return results


def _print_report(results: dict) -> None:
    print(f"{'Model':<18} {'P@10':>8} {'R@10':>8} {'MAP':>8} {'nDCG@10':>8}")
    for name in ("Boolean", "TF-IDF", "BM25", "BM25+expansion"):
        row = results[name]
        print(
            f"{name:<18} {row['precision']:8.3f} {row['recall']:8.3f} "
            f"{row['map']:8.3f} {row['ndcg']:8.3f}"
        )
    print("\nFailed BM25 queries (top hit not relevant, or no highly relevant hit in the top 10):")
    index = results["_index"]
    qrels = results["_qrels"]
    failures = results["BM25"]["failures"]
    if not failures:
        print("  none")
        return
    for failure in failures:
        top = failure["top_doc"]
        title = index.documents[top]["title"] if top in index.documents else "(no result)"
        highly = [
            f"{doc_id} ({index.documents[doc_id]['title']})"
            for doc_id, grade in qrels.get(failure["query_id"], {}).items()
            if grade >= 2
        ]
        print(f"  {failure['query_id']}: {failure['text']}")
        print(f"    top: {title} (grade {failure['top_grade']})")
        print(f"    highly relevant: {'; '.join(highly[:4])}")
    print()
    print("Error analysis")
    print(
        "Q25, addis ababa airport stopover. BM25 ranks Northern Ethiopia first. "
        "That page is a travel warning, and it uses stopover for Mekele, Debre Marqos, "
        "and Dessie as bus-transfer towns. The stem stopov therefore matches three times. "
        "The Bole airport stopover guide is shorter and lands at rank 3, with the airport "
        "pages further down. Pseudo-relevance feedback then adds jimma, tigrai, and updat "
        "from those warning pages, and the guide falls to rank 6."
    )
    print(
        "Expansion also hurts queries that BM25 already answers. "
        "For the coffee ceremony, feedback adds dish, spice, and jimma, and the ceremony "
        "page drops out of the top 10. For Timkat, feedback adds meskel and squar, so "
        "Meskel Square, a different festival, becomes the first hit. For the visa query, "
        "feedback adds birr, dollar, and car, and currency pages outrank the visa pages. "
        "For Lake Tana, feedback adds bahir and dar, which pulls the city pages up and "
        "lowers nDCG even though the lake pages stay in the list."
    )
    print(
        "Expansion helps when a name has another spelling or the feedback terms stay "
        "on topic. Lalibela, injera, Konso, and Dallol all gain nDCG. Axum and Simien "
        "are already covered by the synonym list."
    )
    print(
        "Boolean retrieval returns matching documents in identifier order, with no score. "
        "Recall stays reasonable because the matching set is large, but the first ten "
        "are often city or practical pages that merely contain every query word. "
        "That is why coffee, food, Timkat, and gelada start on the wrong document."
    )
    print(
        "TF-IDF cosine prefers a short page when a rare term is dense in it. "
        "On injera and ethiopian food, Nature Experience is about 100 terms long and "
        "mentions injera and food, so it ranks above the long Injera, wat, and cuisine "
        "articles. BM25 still puts those articles first."
    )


if __name__ == "__main__":
    _print_report(compare())
