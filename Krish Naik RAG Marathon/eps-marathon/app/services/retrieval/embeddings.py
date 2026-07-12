import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768


_active_model: None
_model_type: str | None = None


def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None"""


def _load_fallback():
    return

def _init():
    return

def get_embedding_dim() -> int:
    """Retrun the Vector dimensions for the acctive model. Call after _init()."""
    return

def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Call after _init()."""
    return

def embed_query(query: str) -> list[float]:
    """Embed a single query. Call after _init()."""
    return

def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    return






















