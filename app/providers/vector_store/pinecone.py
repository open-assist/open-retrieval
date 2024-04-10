import os
import pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
assert PINECONE_API_KEY is not None
assert PINECONE_ENVIRONMENT is not None


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))


def get_vector_store(index_name: str):
    """Create a Pinecone vector store.

    Args:
        index_name (str): The name of the Pinecone index.

    Returns:
        PineconeVectorStore: A Pinecone vector store instance.
    """
    pinecone.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
    )
    return PineconeVectorStore(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
        index_name=index_name,
    )
