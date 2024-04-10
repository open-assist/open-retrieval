import os
from llama_index.core import Settings


embedding = os.environ.get("EMBEDDING")
assert embedding is not None


def set_embed_model():
    """Set the embedding model based on the EMBEDDING environment variable.

    Raises:
        ValueError: If the EMBEDDING environment variable is not set to a supported value.
    """
    match (embedding):
        case "voyage":
            from .voyage import get_embed_model

            Settings.embed_model = get_embed_model()
        case _:
            raise ValueError(f"Unsupported embedding: {embedding}")
