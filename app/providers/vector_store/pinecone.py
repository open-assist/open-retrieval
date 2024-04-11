import os
from pinecone import Pinecone, PodSpec, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore


def get_pinecone_vector_store(index_name: str, create: bool = True):
    """Create a Pinecone vector store.

    Args:
        index_name (str): The name of the Pinecone index.

    Returns:
        PineconeVectorStore: A Pinecone vector store instance.
    """
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    assert PINECONE_API_KEY is not None

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if create:
        spec = PodSpec("gcp-starter")
        PINECONE_SPEC = os.environ.get("PINECONE_SPEC")
        if PINECONE_SPEC == "pod":
            environment = os.environ.get("PINECONE_POD_ENVIRONMENT")
            assert environment is not None
            pod_type = os.environ.get("PINECONE_POD_TYPE")
            assert pod_type is not None
            pods = os.environ.get("PINECONE_POD_TYPE")
            assert pods is not None

            spec = PodSpec(environment=environment, pods=int(pods), pod_type=pod_type)
        elif PINECONE_SPEC == "serverless":
            cloud = os.environ.get("PINECONE_SERVERLESS_CLOUD")
            assert cloud is not None
            region = os.environ.get("PINECONE_SERVERLESS_REGION")
            assert region is not None

            spec = ServerlessSpec(cloud=cloud, region=region)

        EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))
        pc.create_index(name=index_name, dimension=EMBEDDING_DIMENSION, spec=spec)
    return PineconeVectorStore(
        pc.Index(index_name),
    )
