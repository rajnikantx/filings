from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str] | None = None


class IngestionRequest(BaseModel):
    directory: str = "data/raw_filings/"


class IngestionResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_exists: bool
