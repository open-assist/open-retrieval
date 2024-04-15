import os
from llama_index.core import VectorStoreIndex
from ...dependencies import logger
from .pinecone import get_pinecone_vector_store, delete_pinecone_index
from .postgres import get_postgres_vector_store


def _get_vector_store_name():
    vector_store = os.environ.get("VECTOR_STORE")
    assert vector_store is not None
    return vector_store


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

    vector_store = _get_vector_store_name()

    match vector_store:
        case "pinecone":
            return get_pinecone_vector_store(index_name, create)
        case "postgres":
            return get_postgres_vector_store(index_name, create)
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


def delete_index(index_name: str):
    """Delete a vector store index.

    Args:
        index_name (str): The name of the index.

    Raises:
        ValueError: If the vector store is not supported.
    """
    vector_store = _get_vector_store_name

    match vector_store:
        case "pinecone":
            return delete_pinecone_index(index_name)
        case "postgres":
            return
        case _:
            raise ValueError(f"Unsupported vector store: {vector_store}")
