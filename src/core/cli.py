import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEC Filing RAG Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingestion_cmd = sub.add_parser("ingestion", help="Run the ingestion pipeline")
    ingestion_cmd.add_argument(
        "--directory",
        default="data/raw_filings/",
        help="Directory containing raw PDF filings",
    )

    query_cmd = sub.add_parser("query", help="Query the system")
    query_cmd.add_argument("--query", "-q", required=True, help="The question to ask")
    query_cmd.add_argument(
        "--top-k", type=int, default=5, help="Number of chunks to retrieve"
    )

    return parser.parse_args()
