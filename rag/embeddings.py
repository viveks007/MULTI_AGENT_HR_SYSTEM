import hashlib
from typing import List

_model = None


def _load_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError(
                "sentence_transformers is required for real embeddings. "
                "Install it with `pip install sentence-transformers`.") from exc
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model


def get_embedding(text: str) -> List[float]:
    try:
        model = _load_model()
        return model.encode(text).tolist()
    except Exception:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest]
