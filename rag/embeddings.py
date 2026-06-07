import hashlib
import math
import re
from typing import List

_model = None
_model_load_failed = False
FALLBACK_EMBEDDING_DIM = 512
_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9']+\b")


def _load_model():
    global _model, _model_load_failed
    if _model_load_failed:
        raise ImportError("sentence_transformers is not available")

    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            _model_load_failed = True
            raise ImportError(
                "sentence_transformers is required for real embeddings. "
                "Install it with `pip install sentence-transformers`."
            ) from exc
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model


def _stable_token_index(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False) % FALLBACK_EMBEDDING_DIM


def _fallback_embedding(text: str) -> List[float]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return [0.0] * FALLBACK_EMBEDDING_DIM

    vector = [0.0] * FALLBACK_EMBEDDING_DIM
    for token in tokens:
        vector[_stable_token_index(token)] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * FALLBACK_EMBEDDING_DIM

    return [value / norm for value in vector]


def get_embedding(text: str) -> List[float]:
    try:
        model = _load_model()
        embedding = model.encode(text)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)
    except Exception:
        return _fallback_embedding(text)
