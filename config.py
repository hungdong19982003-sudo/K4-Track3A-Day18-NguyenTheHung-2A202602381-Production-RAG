"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & LLM Provider ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Tự động nhận diện provider: ưu tiên Gemini nếu có GEMINI_API_KEY hoặc nếu OPENAI_API_KEY bắt đầu bằng AIza
if (GEMINI_API_KEY and not GEMINI_API_KEY.startswith("AIza...")) or OPENAI_API_KEY.startswith("AIza"):
    LLM_PROVIDER = "gemini"
    LLM_API_KEY = GEMINI_API_KEY if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("AIza...") else OPENAI_API_KEY
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    # Thiết lập biến môi trường để thư viện OpenAI/LangChain mặc định tự động dùng Gemini
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
    LLM_PROVIDER = "openai"
    LLM_API_KEY = OPENAI_API_KEY
    LLM_BASE_URL = None
    LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
else:
    LLM_PROVIDER = "gemini" if GEMINI_API_KEY else "openai"
    LLM_API_KEY = GEMINI_API_KEY or OPENAI_API_KEY
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" if GEMINI_API_KEY else None
    LLM_MODEL = "gemini-3.1-flash-lite" if GEMINI_API_KEY else "gpt-4o-mini"


def get_llm_client():
    """Tạo client OpenAI tương thích (hoạt động với cả OpenAI và Gemini)."""
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-...") or LLM_API_KEY.startswith("AIza..."):
        return None
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# --- Qdrant ---
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
