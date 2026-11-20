from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:
    """Create, persist, load, and search a FAISS vector database."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        store_path: str = "data/vector_store"
    ):
        self.store_path = Path(store_path)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.database = None

    def create_store(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("Cannot create a vector store from empty documents.")

        self.database = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        self.store_path.mkdir(parents=True, exist_ok=True)

        self.database.save_local(str(self.store_path))

    def load_store(self) -> None:
        if not self.store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found: {self.store_path}"
            )

        index_file = self.store_path / "index.faiss"

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_file}"
            )

        self.database = FAISS.load_local(
            str(self.store_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Document]:
        if self.database is None:
            raise RuntimeError("Vector store is not initialized.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        return self.database.similarity_search(query, k=k)
