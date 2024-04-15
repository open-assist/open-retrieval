import os
from llama_index.vector_stores.postgres import PGVectorStore


def get_postgres_vector_store(index_name: str, create: bool = True):
    PG_HOST = os.environ.get("PG_HOST", "localhost")
    PG_PORT = os.environ.get("PG_PORT", "5432")
    PG_DATABASE = os.environ.get("PG_DB", "postgres")
    PG_USER = os.environ.get("PG_USER", "postgres")
    PG_PASSWORD = os.environ.get("PG_PASSWORD", "postgres")
    EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))

    return PGVectorStore.from_params(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        table_name=index_name,
        embed_dim=EMBEDDING_DIMENSION,
    )
