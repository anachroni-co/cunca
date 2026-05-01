"""CUNCA Corpus processing pipeline.

Implements:
  - C4-style quality filters  (language heuristics, length, repetition)
  - MinHash-LSH near-duplicate detection (pure Python, no datasketch dep)
  - CUNCACorpusProcessor: orchestrates filter → dedup → shard pipeline

Target: 250B tokens (gl + pt) per the CUNCA Memoria Técnica (Anexo VIII).

Usage:
    from data.cunca_corpus import CUNCACorpusProcessor, C4Filter

    filt = C4Filter(min_words=50, max_word_ratio=0.2)
    ok, reason = filt.apply("Texto de exemplo en galego...")
    assert ok

    proc = CUNCACorpusProcessor(output_dir="data/processed/cunca")
    stats = proc.process_file("data/raw/gl/corpus.txt")
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# C4-style quality filters
# ---------------------------------------------------------------------------

_GALICIAN_STOPS = frozenset(
    "o a os as un unha uns unhas de do da dos das en por para con sen e ou pero "
    "mais non si que non é son foi foi era eran ten teñen".split()
)
_PORTUGUESE_STOPS = frozenset(
    "o a os as um uma uns umas de do da dos das em por para com sem e ou mas "
    "mais não sim que é são foi eram tem têm".split()
)
_SPANISH_STOPS = frozenset(
    "el la los las un una unos unas de del en por para con sin y o pero "
    "más no sí que es son fue eran tiene tienen".split()
)

_STOP_SETS = {"gl": _GALICIAN_STOPS, "pt": _PORTUGUESE_STOPS, "es": _SPANISH_STOPS}

_BULLET_RE = re.compile(r"^[•\-\*]\s+", re.MULTILINE)
_LOREM_RE = re.compile(r"lorem ipsum", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,:;])\1{3,}")


@dataclass
class FilterResult:
    passed: bool
    reason: str = ""


class C4Filter:
    """C4-style document quality filter.

    Applies heuristics from the C4 paper (Raffel et al. 2020) adapted for
    Galician / Portuguese:
      1. Minimum word count
      2. Max ratio of words ending with terminal punctuation
      3. Minimum fraction of stop words
      4. No lorem ipsum
      5. No excessive bullet lines (scrape artifacts)
      6. No excessive repeated punctuation
      7. Max URL density
    """

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100_000,
        min_stop_word_ratio: float = 0.05,
        max_terminal_ratio: float = 0.30,
        max_url_density: float = 0.10,
        lang: str = "gl",
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.min_stop_word_ratio = min_stop_word_ratio
        self.max_terminal_ratio = max_terminal_ratio
        self.max_url_density = max_url_density
        self.stop_words = _STOP_SETS.get(lang, _GALICIAN_STOPS)

    def apply(self, text: str) -> tuple[bool, str]:
        """Return (passed, reason). reason is '' when passed."""
        text = text.strip()
        if not text:
            return False, "empty"

        words = text.split()
        n = len(words)

        if n < self.min_words:
            return False, f"too_short({n}<{self.min_words})"
        if n > self.max_words:
            return False, f"too_long({n}>{self.max_words})"

        if _LOREM_RE.search(text):
            return False, "lorem_ipsum"

        # Stop word ratio
        stop_count = sum(1 for w in words if w.lower().rstrip(".,;:!?") in self.stop_words)
        if stop_count / n < self.min_stop_word_ratio:
            return False, f"low_stop_words({stop_count/n:.3f}<{self.min_stop_word_ratio})"

        # Terminal punctuation ratio (C4: <30% of lines end with terminal punct)
        # Only meaningful for multi-paragraph documents (≥5 lines)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 5:
            terminal = sum(1 for l in lines if l and l[-1] in ".!?")
            if terminal / len(lines) > self.max_terminal_ratio:
                return False, f"high_terminal_ratio({terminal/len(lines):.2f})"

        # URL density
        urls = _URL_RE.findall(text)
        url_chars = sum(len(u) for u in urls)
        if len(text) > 0 and url_chars / len(text) > self.max_url_density:
            return False, f"high_url_density({url_chars/len(text):.3f})"

        # Repeated punctuation
        if _REPEATED_PUNCT_RE.search(text):
            return False, "repeated_punctuation"

        return True, ""


# ---------------------------------------------------------------------------
# MinHash-LSH near-duplicate detection
# ---------------------------------------------------------------------------

_MINHASH_PERMS = 128    # number of hash functions
_LSH_BANDS = 16
_LSH_ROWS = _MINHASH_PERMS // _LSH_BANDS   # 8 rows per band

# Pre-generated (a, b) pairs for MinHash universal hash functions
_A, _B = zip(*[
    (struct.unpack("I", hashlib.sha256(f"a{i}".encode()).digest()[:4])[0],
     struct.unpack("I", hashlib.sha256(f"b{i}".encode()).digest()[:4])[0])
    for i in range(_MINHASH_PERMS)
])
_LARGE_PRIME = (1 << 31) - 1


def _shingles(text: str, k: int = 5) -> set[int]:
    """Character k-shingles hashed to integers."""
    text = text.lower()
    return {
        struct.unpack("I", hashlib.md5(text[i:i + k].encode()).digest()[:4])[0]
        for i in range(max(1, len(text) - k + 1))
    }


def _minhash(shingles: set[int]) -> list[int]:
    """Compute MinHash signature (list of _MINHASH_PERMS hash values)."""
    sig = [_LARGE_PRIME] * _MINHASH_PERMS
    for sh in shingles:
        for i, (a, b) in enumerate(zip(_A, _B)):
            h = (a * sh + b) % _LARGE_PRIME
            if h < sig[i]:
                sig[i] = h
    return sig


def _lsh_bands(sig: list[int]) -> list[tuple]:
    """Split MinHash signature into LSH band tuples."""
    bands = []
    for band in range(_LSH_BANDS):
        start = band * _LSH_ROWS
        end = start + _LSH_ROWS
        bands.append((band, tuple(sig[start:end])))
    return bands


class MinHashLSH:
    """Near-duplicate detector using MinHash + LSH.

    Jaccard similarity threshold ≈ (1/bands)^(1/rows).
    With _LSH_BANDS=16, _LSH_ROWS=8: threshold ≈ 0.50 Jaccard similarity.
    """

    def __init__(self) -> None:
        self._buckets: dict[tuple, set[str]] = {}
        self._seen: set[str] = set()

    def is_duplicate(self, doc_id: str, text: str) -> bool:
        """Return True if text is a near-duplicate of a previously seen document."""
        shingles = _shingles(text)
        if not shingles:
            return False
        sig = _minhash(shingles)
        bands = _lsh_bands(sig)

        for band_key in bands:
            if band_key in self._buckets:
                candidates = self._buckets[band_key]
                if candidates - {doc_id}:
                    return True

        # Not a duplicate — add to index
        for band_key in bands:
            self._buckets.setdefault(band_key, set()).add(doc_id)
        self._seen.add(doc_id)
        return False

    def __len__(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# Corpus processor
# ---------------------------------------------------------------------------

@dataclass
class ProcessingStats:
    total_docs: int = 0
    passed_filter: int = 0
    deduplicated: int = 0
    written: int = 0
    filter_reasons: dict = field(default_factory=dict)

    @property
    def kept_ratio(self) -> float:
        return self.written / max(self.total_docs, 1)


class CUNCACorpusProcessor:
    """Applies C4 filtering + MinHash-LSH deduplication to text corpora.

    Output format: one document per line (double-newline as doc separator
    is compatible with data.loader.SlimDataset when read via data.preprocess).

    Args:
        output_dir:     Directory for processed output files.
        lang:           Source language for C4 stop-word lists ('gl','pt','es').
        min_words:      Minimum word count per document.
        dedup:          Enable MinHash-LSH deduplication.
        shard_size:     Docs per output shard file.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed/cunca",
        lang: str = "gl",
        min_words: int = 50,
        dedup: bool = True,
        shard_size: int = 50_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._filter = C4Filter(min_words=min_words, lang=lang)
        self._lsh = MinHashLSH() if dedup else None
        self.shard_size = shard_size

    def _iter_docs(self, path: Path) -> Iterator[str]:
        """Yield documents from a file (double-newline separated or one-per-line)."""
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = text.split("\n\n")
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                yield chunk

    def process_file(self, input_path: str | Path) -> ProcessingStats:
        """Filter and dedup a single input file, write to output_dir."""
        input_path = Path(input_path)
        stats = ProcessingStats()
        shard_idx = 0
        buffer: list[str] = []

        for doc_idx, doc in enumerate(self._iter_docs(input_path)):
            stats.total_docs += 1
            passed, reason = self._filter.apply(doc)
            if not passed:
                stats.filter_reasons[reason] = stats.filter_reasons.get(reason, 0) + 1
                continue
            stats.passed_filter += 1

            if self._lsh is not None:
                doc_id = f"{input_path.stem}_{doc_idx}"
                if self._lsh.is_duplicate(doc_id, doc[:1000]):
                    stats.deduplicated += 1
                    continue

            buffer.append(doc)
            stats.written += 1

            if len(buffer) >= self.shard_size:
                self._write_shard(buffer, input_path.stem, shard_idx)
                shard_idx += 1
                buffer = []

        if buffer:
            self._write_shard(buffer, input_path.stem, shard_idx)

        logger.info(
            "Processed %s: total=%d kept=%d (%.1f%%) deduped=%d",
            input_path.name, stats.total_docs, stats.written,
            100 * stats.kept_ratio, stats.deduplicated,
        )
        return stats

    def _write_shard(self, docs: list[str], stem: str, idx: int) -> None:
        out = self.output_dir / f"{stem}_shard_{idx:04d}.txt"
        out.write_text("\n\n".join(docs) + "\n", encoding="utf-8")


__all__ = [
    "C4Filter",
    "FilterResult",
    "MinHashLSH",
    "CUNCACorpusProcessor",
    "ProcessingStats",
]
