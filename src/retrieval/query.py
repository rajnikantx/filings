from src.ingestion.encoder import Encoder
from src.ingestion.vectorizer import Vectorizer

if __name__ == "__main__":
    encoder = Encoder()
    vec = Vectorizer(encoder)

    # Verify collection exists before searching
    if not vec._client.collection_exists(vec._collection):
        print(f"ERROR: Collection '{vec._collection}' not found.")
        print("Run 'python ingest.py' first to create and populate the collection.")
        exit(1)

    hits = vec.search(
        query="Where is Tesla located?",
        filters={"ticker": "TSLA"},
        limit=2,
    )

    print(f"Found {len(hits)} hit(s):\n")
    for hit in hits:
        print(f"Score: {hit.score:.4f}")
        print(f"Content: {hit.payload.get('page_content')}")
        print(f"Metadata: {hit.payload}")
        print("-" * 40)