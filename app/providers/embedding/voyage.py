import os
from llama_index.embeddings.voyageai import VoyageEmbedding

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL")
assert VOYAGE_API_KEY is not None
assert VOYAGE_MODEL is not None


def get_embed_model():
    """Get the embedding model.

    Returns:
        VoyageEmbedding: The embedding model using the specified Voyage model and API key.
    """
    return VoyageEmbedding(model_name=VOYAGE_MODEL, voyage_api_key=VOYAGE_API_KEY)
