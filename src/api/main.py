from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from loguru import logger

from src.api.models import (
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
)
from src.config import settings
from src.indexing.vector_store import VectorStore
from src.inference.chunks_retrieval import ChunkRetrieval, ChunkRetrievalError
from src.inference.context_build import Context
from src.inference.generate_answer import Generation
from src.inference.query_enhancement import QueryEnhancement


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API server")
    yield
    logger.info("Shutting down API server")


app = FastAPI(
    title="SEC Filing RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.exception_handler(ChunkRetrievalError)
async def chunk_retrieval_error_handler(request, exc):
    raise HTTPException(status_code=502, detail=str(exc))


@app.exception_handler(Exception)
async def global_error_handler(request, exc):
    logger.error("Unhandled error: {}", exc)
    raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    vs = VectorStore()
    exists = await vs._client.collection_exists(settings.QDRANT_COLLECTION)
    if not exists:
        raise HTTPException(
            status_code=400,
            detail="No data ingested. Run POST /ingestion first.",
        )

    collection_info = await vs._client.get_collection(settings.QDRANT_COLLECTION)
    if collection_info.points_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Collection is empty. Run POST /ingestion first.",
        )

    enhancer = QueryEnhancement()
    retriever = ChunkRetrieval()
    context_builder = Context()
    generator = Generation()

    try:
        rewritten = await enhancer.query_rewrite(request.query)
    except Exception as e:
        logger.warning("Query rewrite failed, using original: {}", e)
        rewritten = request.query

    try:
        results = await retriever.search(
            rewritten, limit=request.top_k, filters=request.filters
        )
    except ChunkRetrievalError:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}")

    context = context_builder.build_context(results)

    async def stream():
        async for token in generator.generate_answer(context, request.query):
            yield token

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/ingestion", response_model=IngestionResponse)
async def ingestion_endpoint(request: IngestionRequest):
    import asyncio

    from src.main import ingestion

    asyncio.create_task(ingestion(directory=request.directory))

    return IngestionResponse(
        status="started",
        message=f"Ingestion started for {request.directory}",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    qdrant_connected = False
    collection_exists = False

    try:
        vs = VectorStore()
        collection_exists = await vs._client.collection_exists(
            settings.QDRANT_COLLECTION
        )
        qdrant_connected = True
    except Exception as e:
        logger.warning("Health check failed: {}", e)

    return HealthResponse(
        status="ok" if qdrant_connected else "degraded",
        qdrant_connected=qdrant_connected,
        collection_exists=collection_exists,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="localhost", port=8000, reload=True)
