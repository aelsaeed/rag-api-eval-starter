from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    doc_id: str = Field(..., description="Document identifier")
    chunks: int = Field(..., description="Number of chunks stored")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "How does hybrid retrieval work?",
                }
            ]
        }
    )


class Citation(BaseModel):
    doc_id: str | None
    chunk_id: str | None
    snippet: str | None
    score: float
    dense_score: float
    keyword_score: float
    source: str | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": (
                        "Answer (extractive):\n"
                        "- Hybrid retrieval blends dense and keyword scores."
                    ),
                    "citations": [
                        {
                            "doc_id": "example-doc",
                            "chunk_id": "example-doc-0",
                            "snippet": (
                                "The API blends dense vector similarity "
                                "with keyword overlap."
                            ),
                            "score": 0.83,
                            "dense_score": 0.92,
                            "keyword_score": 0.65,
                            "source": "platform_overview.md",
                        }
                    ],
                }
            ]
        }
    )


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None
