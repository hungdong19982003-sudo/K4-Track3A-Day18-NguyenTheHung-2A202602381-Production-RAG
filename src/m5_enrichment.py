from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


import json as _json
import re as _re
from config import LLM_MODEL, get_llm_client

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "enrichment_cache.json")
_ENRICH_CACHE = None

def _get_cache():
    global _ENRICH_CACHE
    if _ENRICH_CACHE is None:
        _ENRICH_CACHE = {}
        if os.path.exists(_CACHE_FILE):
            try:
                with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                    _ENRICH_CACHE = _json.load(f)
            except Exception:
                _ENRICH_CACHE = {}
    return _ENRICH_CACHE

def _save_cache():
    if _ENRICH_CACHE:
        try:
            os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                _json.dump(_ENRICH_CACHE, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def _call_llm_with_fallback(messages: list[dict], max_tokens: int = 200) -> str | None:
    client = get_llm_client()
    if not client:
        return None
    models_to_try = [LLM_MODEL, "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]
    seen = set()
    unique_models = [m for m in models_to_try if not (m in seen or seen.add(m))]
    for m in unique_models:
        try:
            resp = client.chat.completions.create(model=m, messages=messages, max_tokens=max_tokens)
            content = resp.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "ResourceExhausted" in str(e):
                continue
    return None

def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_key = f"summary:{text_hash}"
    cache = _get_cache()
    if cache_key in cache:
        return cache[cache_key]

    content = _call_llm_with_fallback([
        {"role": "system", "content": "Tóm tắt đoạn văn sau trong 1 câu thật ngắn gọn bằng tiếng Việt (dưới 15 từ)."},
        {"role": "user", "content": text},
    ], max_tokens=80)
    if content and len(content) <= len(text) * 1.8:
        cache[cache_key] = content
        _save_cache()
        return content

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    fallback = sentences[0] if sentences else text
    if len(fallback) > len(text) * 1.5:
        fallback = text[:len(text) // 2]
    return fallback


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_key = f"hyqa:{text_hash}:{n_questions}"
    cache = _get_cache()
    if cache_key in cache:
        return cache[cache_key]

    content = _call_llm_with_fallback([
        {"role": "system", "content": f"Dựa trên đoạn văn, hãy tạo {n_questions} câu hỏi bằng tiếng Việt có kết thúc bằng dấu chấm hỏi (?). Mỗi câu hỏi nằm trên một dòng riêng biệt."},
        {"role": "user", "content": text},
    ], max_tokens=150)
    if content:
        questions = content.split("\n")
        cleaned = [q.strip().lstrip("0123456789.-*#) ") for q in questions if q.strip()]
        valid_qs = [q if q.endswith("?") else f"{q}?" for q in cleaned if len(q) > 5]
        if valid_qs:
            res = valid_qs[:n_questions]
            cache[cache_key] = res
            _save_cache()
            return res

    sentences = [s.strip() for s in _re.split(r'[.!?\n]', text) if len(s.strip()) > 5]
    if not sentences:
        sentences = [text]
    fallback_qs = []
    for s in sentences[:n_questions]:
        clean = s.rstrip('.!?')
        fallback_qs.append(f"{clean} như thế nào?")
    return fallback_qs


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_key = f"contextual:{text_hash}:{document_title}"
    cache = _get_cache()
    if cache_key in cache:
        return cache[cache_key]

    content = _call_llm_with_fallback([
        {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về đúng 1 câu duy nhất."},
        {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=80)
    if content:
        res = f"{content}\n\n{text}"
        cache[cache_key] = res
        _save_cache()
        return res

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_key = f"metadata:{text_hash}"
    cache = _get_cache()
    if cache_key in cache:
        return cache[cache_key]

    content = _call_llm_with_fallback([
        {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về CHỈ duy nhất 1 JSON hợp lệ: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi"}'},
        {"role": "user", "content": text},
    ], max_tokens=150)
    if content:
        try:
            if "```" in content:
                content = _re.sub(r'```(?:json)?', '', content).strip()
            res = _json.loads(content)
            cache[cache_key] = res
            _save_cache()
            return res
        except Exception:
            pass

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache = _get_cache()
    if text_hash in cache:
        return cache[text_hash]

    models_to_try = [LLM_MODEL, "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]
    seen = set()
    unique_models = [m for m in models_to_try if not (m in seen or seen.add(m))]
    client = get_llm_client()
    if client:
        for m in unique_models:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": """Phân tích đoạn văn và trả về CHỈ duy nhất 1 JSON hợp lệ có định dạng:
{
  "summary": "tóm tắt 1-2 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả vị trí và chủ đề đoạn văn trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi"}
}"""},
                        {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                    ],
                    max_tokens=400,
                )
                content = resp.choices[0].message.content.strip()
                if "```" in content:
                    content = _re.sub(r'```(?:json)?', '', content).strip()
                parsed = _json.loads(content)
                cache[text_hash] = parsed
                _save_cache()
                return parsed
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "ResourceExhausted" in str(e):
                    continue
                print(f"  ⚠️  Enrichment API failed on {m}: {e}")

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    fallback = {
        "summary": sentences[0] if sentences else text,
        "questions": [f"{s.rstrip('.')}?" for s in sentences[:3]],
        "context": f"Tài liệu: {source}" if source else "",
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }
    return fallback


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
