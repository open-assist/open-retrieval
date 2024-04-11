import os
from llama_index.core import VectorStoreIndex
from ...dependencies import logger
from .pinecone import get_pinecone_vector_store


def get_vector_store(index_name: str, create: bool = True):
    """Get a vector store instance.

    Args:
        index_name (str): The name of the index.

    Returns:
        The vector store instance.

    Raises:
        ValueError: If the vector store is not supported.
    """
    logger.info(f"[{__name__}] index name: {index_name}")

    vector_store = os.environ.get("VECTOR_STORE")
    assert vector_store is not None

    match vector_store:
        case "pinecone":
            return get_pinecone_vector_store(index_name, create)
        case _:
            raise ValueError(f"Unsupported vector store: {vector_store}")


def get_vector_store_index(index_name: str):
    """Get a VectorStoreIndex instance from a vector store.

    Args:
        index_name (str): The name of the index.

    Returns:
        VectorStoreIndex: The VectorStoreIndex instance.
    """
    vector_store = get_vector_store(index_name, False)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
