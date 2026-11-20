from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class DocumentProcessor:
    """Load and split PDF and TXT documents."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_document(self, file_path: str) -> List[Document]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {path.suffix}"
            )

        documents = []

        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8-sig")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name}
                    )
                )

        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": path.name,
                                "page": page_number
                            }
                        )
                    )

        return documents

    def load_directory(self, directory_path: str) -> List[Document]:
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(
                f"Directory not found: {directory}"
            )

        documents = []

        for file_path in sorted(directory.iterdir()):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ):
                documents.extend(self.load_document(str(file_path)))

        return documents

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        return self.splitter.split_documents(documents)
