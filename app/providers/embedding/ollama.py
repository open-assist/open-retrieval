import os
from llama_index.embeddings.ollama import OllamaEmbedding

OLLAMA_URL = os.environ.get("OLLAMA_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
assert OLLAMA_URL is not None
assert EMBEDDING_MODEL is not None


def get_ollama_embed_model():
    """Get the embedding model.

    Returns:
        OllamaEmbedding: The embedding model using the specified model name and base url.
    """
    return OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_URL)
