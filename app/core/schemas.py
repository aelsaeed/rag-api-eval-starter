from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    doc_id: str = Field(..., description="Document identifier")
    chunks: int = Field(..., description="Number of chunks stored")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1_000, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=20)
    strategy: Literal["dense", "lexical", "hybrid"] | None = None

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "question": "How does hybrid retrieval work?",
                }
            ]
        },
    )


class Citation(BaseModel):
    doc_id: str | None
    chunk_id: str | None
    snippet: str | None
    score: float = Field(ge=0.0, le=1.0)
    evidence_score: float = Field(ge=0.0, le=1.0)
    dense_score: float | None
    keyword_score: float
    dense_rank: int | None
    keyword_rank: int | None
    source: str | None


class RetrievalMetadata(BaseModel):
    strategy: Literal["dense", "lexical", "hybrid"]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    dense_candidates: int = Field(ge=0)
    lexical_candidates: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval: RetrievalMetadata

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": (
                        "Answer (extractive):\n- Hybrid retrieval blends dense and keyword scores."
                    ),
                    "citations": [
                        {
                            "doc_id": "example-doc",
                            "chunk_id": "example-doc-0",
                            "snippet": (
                                "The API blends dense vector similarity with keyword overlap."
                            ),
                            "score": 0.83,
                            "evidence_score": 0.89,
                            "dense_score": 0.92,
                            "keyword_score": 0.65,
                            "dense_rank": 1,
                            "keyword_rank": 1,
                            "source": "platform_overview.md",
                        }
                    ],
                    "retrieval": {
                        "strategy": "hybrid",
                        "confidence": 0.89,
                        "abstained": False,
                        "dense_candidates": 12,
                        "lexical_candidates": 4,
                        "latency_ms": 8.4,
                    },
                }
            ]
        }
    )


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None
