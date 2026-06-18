from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str
    LLAMA_CLOUD_API_KEY: str
    VISION_AGENT_API_KEY: str
    METADATA_MODEL: str = "gpt-4o-mini"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RAW_FILINGS_DIR: Path = BASE_DIR / "data" / "raw_filings"
    PARSED_CONTENT_DIR: Path = BASE_DIR / "outputs" / "content"
    INTRO_DIR: Path = BASE_DIR / "outputs" / "intro"
    PARENT_CHILD_PATH: Path = BASE_DIR / "outputs" / "parent_child"
    CHUNK_DIR: Path = BASE_DIR / "outputs" / "chunks"

    CHUNK_SIZE: int = 700

    EMBEDDING_MODEL: str = "FinLang/finance-embeddings-investopedia"
    EMBEDDING_DIM: int = 768
    EMBEDDING_QUERY_INSTRUCTION: str = ""
    QDRANT_COLLECTION: str = "sec_filings"
    QDRANT_URL: str = "/home/rajnikant/Github/filings/qdrant_storage"

settings = Settings()