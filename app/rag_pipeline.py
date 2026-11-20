import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

from app.document_processor import DocumentProcessor
from app.vector_store import VectorStore


load_dotenv()


class RAGPipeline:
    """Retrieve relevant documents and generate answers using Groq LLM."""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from the .env file.")

        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-20b"

    def ingest_documents(self, directory: str = "data/documents") -> int:
        documents = self.processor.load_directory(directory)

        if not documents:
            raise ValueError("No supported documents found.")

        chunks = self.processor.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks could be created.")

        self.vector_store.create_store(chunks)
        return len(chunks)

    def load_existing_store(self) -> None:
        self.vector_store.load_store()

    def ask(self, question: str, k: int = 3) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        results = self.vector_store.search(question, k=k)

        if not results:
            return {
                "answer": "I could not find relevant information in the documents.",
                "sources": []
            }

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in results
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise knowledge assistant. "
                        "Answer questions using only the provided document context. "
                        "If the answer is not available in the context, clearly say so. "
                        "Do not invent facts. Give clear and concise answers."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Document context:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
            temperature=0.2,
            max_tokens=512
        )

        answer = response.choices[0].message.content or ""

        sources: List[Dict[str, Any]] = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page")
            }
            for doc in results
        ]

        return {
            "answer": answer,
            "sources": sources
        }
