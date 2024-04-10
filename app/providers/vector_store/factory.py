import os
from llama_index.core import VectorStoreIndex


async def get_vector_store(index_name: str):
    """Get a vector store instance.

    Args:
        index_name (str): The name of the index.

    Returns:
        The vector store instance.

    Raises:
        ValueError: If the vector store is not supported.
    """
    vector_store = os.environ.get("VECTOR_STORE")
    assert vector_store is not None

    match vector_store:
        case "pinecone":
            from providers.vector_store.pinecone import get_vector_store

            return get_vector_store(index_name)
        case _:
            raise ValueError(f"Unsupported vector store: {vector_store}")


async def get_vector_store_index(index_name: str):
    """Get a VectorStoreIndex instance from a vector store.

    Args:
        index_name (str): The name of the index.

    Returns:
        VectorStoreIndex: The VectorStoreIndex instance.
    """
    vector_store = await get_vector_store(index_name=index_name)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
