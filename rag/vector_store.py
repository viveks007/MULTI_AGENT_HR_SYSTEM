"""Simple vector store placeholder."""

class VectorStore:
    def __init__(self):
        self._items = []

    def add(self, vector, meta=None):
        self._items.append((vector, meta))

    def search(self, vector, k=5):
        return []
