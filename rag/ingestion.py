from pathlib import Path
from typing import Dict, List, Optional

try:
    from PyPDF2 import PdfReader
except ImportError as exc:
    raise ImportError(
        "PyPDF2 is required for PDF ingestion. Install it with `pip install PyPDF2`."
    ) from exc

from rag.chunking import chunk_documents
from rag.embeddings import get_embedding
from rag.vector_store import VectorStore

TEXT_EXTENSIONS = {".txt", ".md"}


def load_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages).strip()


def load_text_file(text_path: Path) -> str:
    return text_path.read_text(encoding="utf-8", errors="ignore").strip()


def collect_documents(data_dir: str = "data") -> List[Dict[str, str]]:
    root = Path(data_dir)
    documents: List[Dict[str, str]] = []

    pdf_dir = root / "pdfs"
    faq_dir = root / "faq"

    if pdf_dir.exists():
        for pdf_path in sorted(pdf_dir.glob("**/*.pdf")):
            text = load_pdf_text(pdf_path)
            if text:
                documents.append(
                    {
                        "source": str(pdf_path),
                        "text": text,
                        "type": "pdf",
                    }
                )

    if faq_dir.exists():
        for text_path in sorted(faq_dir.glob("**/*")):
            if text_path.suffix.lower() in TEXT_EXTENSIONS:
                text = load_text_file(text_path)
                if text:
                    documents.append(
                        {
                            "source": str(text_path),
                            "text": text,
                            "type": "faq",
                        }
                    )

    return documents


def ingest(data_dir: str = "data", persist_path: Optional[str] = None) -> VectorStore:
    documents = collect_documents(data_dir)
    if persist_path is None:
        persist_path = str(Path(data_dir) / "vector_store.pkl")

    store = VectorStore()

    for document in documents:
        chunks = chunk_documents(document["text"])
        for chunk_index, chunk_text in enumerate(chunks):
            embedding = get_embedding(chunk_text).tolist()
            store.add_document(
                chunk_text=chunk_text,
                embedding=embedding,
                metadata={
                    "source": document["source"],
                    "type": document["type"],
                    "chunk_index": chunk_index,
                },
            )

    store.save(persist_path)
    return store


def load_index(path: str) -> VectorStore:
    return VectorStore.load(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a vector store from PDF and FAQ data."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root folder containing pdfs/ and faq/",
    )
    parser.add_argument(
        "--persist-path",
        default=None,
        help="Path where the serialized vector store will be saved.",
    )
    args = parser.parse_args()

    vector_store = ingest(data_dir=args.data_dir, persist_path=args.persist_path)
    print(f"Created vector store with {len(vector_store.documents)} chunks.")
    print(
        f"Saved vector store to {args.persist_path or str(Path(args.data_dir) / 'vector_store.pkl')}"
    )
