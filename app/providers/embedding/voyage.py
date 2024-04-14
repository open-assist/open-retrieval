import os
from llama_index.embeddings.voyageai import VoyageEmbedding

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
assert VOYAGE_API_KEY is not None
assert EMBEDDING_MODEL is not None


def get_voyage_embed_model():
    """Get the embedding model.

    Returns:
        VoyageEmbedding: The embedding model using the specified model name and API key.
    """
    return VoyageEmbedding(model_name=EMBEDDING_MODEL, voyage_api_key=VOYAGE_API_KEY)
