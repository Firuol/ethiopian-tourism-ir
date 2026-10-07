"""Text preprocessing for the tourism search engine.

Tokenize the title and body, drop English stop words, and stem with the Porter
stemmer so that related word forms share one index term. The URL is kept on
the document record and is not indexed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus.json"

TOKEN_PATTERN = re.compile(r"[a-z]+")

# English function words removed before indexing.
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves",
}


def tokenize(text: str) -> list[str]:
    """Lowercase the text and keep alphabetic tokens, including place names."""
    text = text.lower().replace("'", "")
    return TOKEN_PATTERN.findall(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


class PorterStemmer:
    """Porter stemmer."""

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        stem = self._step1a(word)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5(stem)
        return stem

    def _cons(self, word: str, index: int) -> bool:
        if word[index] in "aeiou":
            return False
        if word[index] == "y":
            return index == 0 or not self._cons(word, index - 1)
        return True

    def _measure(self, stem: str) -> int:
        measure = 0
        index = 0
        length = len(stem)
        while index < length and self._cons(stem, index):
            index += 1
        while index < length:
            while index < length and not self._cons(stem, index):
                index += 1
            if index >= length:
                return measure
            measure += 1
            while index < length and self._cons(stem, index):
                index += 1
        return measure

    def _has_vowel(self, stem: str) -> bool:
        return any(not self._cons(stem, index) for index in range(len(stem)))

    def _double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and self._cons(word, len(word) - 1)

    def _cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        if (
            self._cons(word, len(word) - 1)
            and not self._cons(word, len(word) - 2)
            and self._cons(word, len(word) - 3)
            and word[-1] not in "wxy"
        ):
            return True
        return False

    def _replace(self, word: str, suffix: str, replacement: str, measure: int | None = None) -> str | None:
        if not word.endswith(suffix):
            return None
        stem = word[: -len(suffix)]
        if measure is not None and self._measure(stem) <= measure:
            return None
        return stem + replacement

    def _step1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return stem + "ee"
            return word
        suffix = ""
        if word.endswith("ed"):
            suffix = "ed"
        elif word.endswith("ing"):
            suffix = "ing"
        if not suffix:
            return word
        stem = word[: -len(suffix)]
        if not self._has_vowel(stem):
            return word
        if stem.endswith(("at", "bl", "iz")):
            return stem + "e"
        if self._double_consonant(stem) and stem[-1] not in "lsz":
            return stem[:-1]
        if self._measure(stem) == 1 and self._cvc(stem):
            return stem + "e"
        return stem

    def _step1c(self, word: str) -> str:
        if word.endswith("y") and self._has_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    def _step2(self, word: str) -> str:
        rules = (
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        )
        for suffix, replacement in rules:
            updated = self._replace(word, suffix, replacement, measure=0)
            if updated is not None:
                return updated
        return word

    def _step3(self, word: str) -> str:
        rules = (
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        )
        for suffix, replacement in rules:
            updated = self._replace(word, suffix, replacement, measure=0)
            if updated is not None:
                return updated
        return word

    def _step4(self, word: str) -> str:
        suffixes = (
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        )
        for suffix in sorted(suffixes, key=len, reverse=True):
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if self._measure(stem) > 1:
                return stem
            return word
        if word.endswith("ion"):
            stem = word[:-3]
            if self._measure(stem) > 1 and stem.endswith(("s", "t")):
                return stem
        return word

    def _step5(self, word: str) -> str:
        if word.endswith("e"):
            stem = word[:-1]
            measure = self._measure(stem)
            if measure > 1 or (measure == 1 and not self._cvc(stem)):
                word = stem
        if self._measure(word) > 1 and self._double_consonant(word) and word.endswith("l"):
            word = word[:-1]
        return word


STEMMER = PorterStemmer()


def stem_tokens(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(token) for token in tokens]


def preprocess(text: str) -> list[str]:
    """Tokenize, remove stop words, and stem."""
    return stem_tokens(remove_stopwords(tokenize(text)))


def load_documents(path: Path | None = None) -> list[dict]:
    """Load corpus records. `text` is the original body; `terms` are indexed terms."""
    corpus_path = path or CORPUS_PATH
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = []
    for document in payload["documents"]:
        title = document["title"]
        body = document["text"]
        documents.append(
            {
                "doc_id": document["doc_id"],
                "title": title,
                "url": document["url"],
                "source": document["source"],
                "topic": document["topic"],
                "text": body,
                "terms": preprocess(f"{title}\n{body}"),
            }
        )
    return documents


def _check_stemmer() -> None:
    expected = {
        "churches": "church",
        "visiting": "visit",
        "visited": "visit",
        "parks": "park",
        "festivals": "festiv",
        "trekking": "trek",
    }
    mismatches = [
        f"{word} -> {STEMMER.stem(word)} (expected {stem})"
        for word, stem in expected.items()
        if STEMMER.stem(word) != stem
    ]
    if mismatches:
        raise SystemExit("Porter stemmer failed checks:\n" + "\n".join(mismatches))


if __name__ == "__main__":
    _check_stemmer()
    sentence = "rock churches and national parks for trekking"
    print("Query terms:", preprocess(sentence))
    documents = load_documents()
    sample = next(document for document in documents if document["title"] == "Lalibela")
    print(f"Loaded {len(documents)} documents")
    print(f"{sample['doc_id']} {sample['title']}")
    print(f"Terms: {len(sample['terms'])}  unique: {len(set(sample['terms']))}")
    print("First terms:", " ".join(sample["terms"][:25]))
