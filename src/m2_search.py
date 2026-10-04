from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(c["text"]).lower().split()
            for c in chunks
        ]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in sorted_indices:
            if scores[idx] > 0 or len(results) == 0:
                doc = self.documents[idx]
                results.append(SearchResult(
                    text=doc["text"],
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))
                if len(results) >= top_k:
                    break
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                try:
                    self._encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                except Exception:
                    self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        dim = self._get_encoder().get_sentence_embedding_dimension()
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        points = [
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]}
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        try:
            response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            points = response.points
        except Exception:
            try:
                points = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
            except Exception:
                points = []

        results = []
        for pt in points:
            payload = pt.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            results.append(SearchResult(
                text=text,
                score=float(pt.score),
                metadata=meta,
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    rrf_scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "metadata": result.metadata,
                    "text": result.text
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [
        SearchResult(
            text=item["text"],
            score=item["score"],
            metadata=item["metadata"],
            method="hybrid"
        )
        for item in sorted_items[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
