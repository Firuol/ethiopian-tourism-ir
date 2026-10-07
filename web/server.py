"""Local search page for the tourism collection.

Serves the HTML interface and answers /api/search with Boolean, TF-IDF, BM25,
or BM25 with query expansion.
"""

from __future__ import annotations

import html
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bm25 import BM25Retriever
from boolean import BooleanRetriever
from inverted_index import InvertedIndex
from query_expansion import ExpandedRetriever
from tfidf import TfidfRetriever

HOST = "127.0.0.1"
PORT = 8765
OPERATORS = {"and", "or", "not"}
WORD = re.compile(r"[A-Za-z']+")


def snippet(text: str, needles: list[str], width: int = 280) -> str:
    clean = re.sub(r"\s+", " ", html.unescape(text or "")).strip()
    if not clean:
        return ""
    lower = clean.lower()
    position = -1
    for needle in needles:
        match = re.search(rf"\b{re.escape(needle.lower())}\b", lower)
        if match:
            position = match.start()
            break
    if position < 0:
        return _window(clean, 0, width)
    start = max(0, position - 80)
    end = min(len(clean), start + width)
    return _window(clean, start, end)


def _window(clean: str, start: int, end: int) -> str:
    if start > 0:
        space = clean.find(" ", start)
        if 0 <= space < end:
            start = space + 1
    if end < len(clean):
        space = clean.rfind(" ", start, end)
        if space > start:
            end = space
    piece = clean[start:end].strip()
    if start > 0:
        piece = "…" + piece
    if end < len(clean):
        piece += "…"
    return piece


def highlight_terms(*texts: str) -> list[str]:
    seen: list[str] = []
    for text in texts:
        for word in WORD.findall(text or ""):
            lower = word.lower()
            if lower in OPERATORS or len(lower) < 3 or lower in seen:
                continue
            seen.append(lower)
    return seen


def public_hit(hit: dict, rank: int, needles: list[str]) -> dict:
    score = hit.get("score")
    return {
        "rank": rank,
        "doc_id": hit["doc_id"],
        "title": html.unescape(hit.get("title") or ""),
        "url": hit.get("url") or "",
        "source": hit.get("source") or "",
        "topic": hit.get("topic") or "",
        "score": round(float(score), 4) if isinstance(score, (int, float)) else None,
        "snippet": snippet(hit.get("text") or "", needles),
    }


class SearchHandler(SimpleHTTPRequestHandler):
    index: InvertedIndex
    models: dict

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/info":
            self._send_json({"documents": self.index.num_docs})
            return
        if path == "/api/search":
            self._search(parse_qs(urlparse(self.path).query))
            return
        super().do_GET()

    def _search(self, params: dict[str, list[str]]) -> None:
        query = (params.get("q") or [""])[0].strip()
        model_name = (params.get("model") or ["bm25"])[0]
        retriever = self.models.get(model_name)
        if not query:
            self._send_json({"error": "Enter a query."}, status=400)
            return
        if retriever is None:
            self._send_json({"error": "Unknown model."}, status=400)
            return

        expanded = None
        every = self.index.num_docs
        if model_name == "boolean":
            hits = retriever.search(query)
        elif model_name == "expansion":
            expanded = retriever.expand(query)
            hits = retriever.search(query, k=every)
        else:
            hits = retriever.search(query, k=every)
        total = len(hits)

        terms = highlight_terms(query, expanded or "")
        self._send_json(
            {
                "query": query,
                "model": model_name,
                "expanded": expanded,
                "total": total,
                "terms": terms,
                "results": [public_hit(hit, rank, terms) for rank, hit in enumerate(hits, start=1)],
            }
        )

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_models(index: InvertedIndex) -> dict:
    bm25 = BM25Retriever(index)
    return {
        "boolean": BooleanRetriever(index),
        "tfidf": TfidfRetriever(index),
        "bm25": bm25,
        "expansion": ExpandedRetriever(bm25),
    }


def main() -> None:
    print("Building the index…", flush=True)
    SearchHandler.index = InvertedIndex.build()
    SearchHandler.models = build_models(SearchHandler.index)
    server = ThreadingHTTPServer((HOST, PORT), SearchHandler)
    print(f"Open http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
