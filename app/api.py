from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import RAGPipeline


app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="Document ingestion and semantic search API",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


@lru_cache(maxsize=1)
def get_pipeline():
    return RAGPipeline()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise AI Knowledge Assistant"
    }


@app.post("/ingest")
def ingest_documents():
    try:
        pipeline = get_pipeline()
        count = pipeline.ingest_documents()

        return {
            "message": "Documents indexed successfully",
            "chunks_indexed": count
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        pipeline = get_pipeline()
        pipeline.load_existing_store()

        result = pipeline.ask(
            question=request.question,
            k=request.k
        )

        return result

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
