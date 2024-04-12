import json
import math
import os
import time
import tiktoken
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import Document
from typing import List

from ..dependencies import get_file_path, get_index_name, get_job_file_path, logger
from ..models.file import FileJobStatus
from ..providers.vector_store.factory import get_vector_store


def _calc_tokens(documents: List[Document]):
    """
    Calculate the total number of tokens in the given documents using the specified encoding.

    Args:
        documents (List[Document]): The list of documents to calculate tokens for.

    Returns:
        int: The total number of tokens in the documents.
    """
    encoding_name = os.environ.get("TIKTOKEN_ENCODING", "cl100k_base")
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = 0
    for d in documents:
        tokens += len(encoding.encode(d.get_text()))
    return tokens


def _parse_code_language(file_type: str):
    """
    Parse the code language based on the file type.

    Args:
        file_type (str): The MIME type of the file.

    Returns:
        str: The corresponding code language for the file type, or None if not recognized.
    """
    match file_type:
        case "application/javascript" | "text/javascript":
            return "javascript"
        case "application/json":
            return "json"
        case "application/sql":
            return "sql"
        case "application/typescript" | "text/x-typescript":
            return "typescript"
        case "application/x-httpd-php" | "text/x-php":
            return "php"
        case "application/x-sh":
            return "bash"
        case "application/xml" | "text/xml":
            return "xml"
        case "text/css":
            return "css"
        case "text/html":
            return "html"
        case "text/x-c":
            return "c"
        case "text/x-csharp":
            return "c-sharp"
        case "text/x-c++":
            return "cpp"
        case "text/x-java" | "text/x-java-source":
            return "java"
        case "text/x-markdown" | "text/markdown":
            return "markdown"
        case "text/x-python" | "text/x-script.python":
            return "python"
        case "text/x-ruby":
            return "ruby"
        case "text/x-yaml" | "text/yaml":
            return "yaml"
        case _:
            return None


def _get_node_parser(file_type: str):
    """
    Get the appropriate node parser based on the file type.

    Args:
        file_type (str): The MIME type of the file.
        embed_model: The embedding model to use for semantic splitting.

    Returns:
        NodeParser: The node parser for the file type, either CodeSplitter for recognized code languages or SemanticSplitterNodeParser for other file types.
    """
    language = _parse_code_language(file_type)
    if language:
        return CodeSplitter(language=language)
    else:
        return SemanticSplitterNodeParser(embed_model=Settings.embed_model)


def index_file(
    org: str,
    file_id: str,
    file_name: str,
    file_type: str,
):
    """
    Index a file and store it in a vector database.

    Args:
        org (str): The organization ID.
        file_name (str): The file name.
        file_type (str): The MIME type of the file.
    """
    logger.info(f"[{__name__}] indexing file({org}/{file_name}) was started")
    job_file = get_job_file_path(org, file_id)
    job_dict = json.loads(job_file.read_text())

    job_dict["status"] = FileJobStatus.RUNNING.value
    job_file.write_text(json.dumps(job_dict))

    try:
        reader = SimpleDirectoryReader(input_files=[str(get_file_path(org, file_name))])
        documents = reader.load_data()
        tokens = _calc_tokens(documents)
        job_dict["tokens"] = tokens

        vector_store = get_vector_store(get_index_name(org, file_id))
        node_parser = _get_node_parser(file_type)
        pipeline = IngestionPipeline(
            transformations=[node_parser, Settings.embed_model],
            vector_store=vector_store,
        )
        pipeline.run(documents=documents)
    except Exception as e:
        logger.error(f"[{__name__}] indexing file({org}/{file_name}) with error: {e}")
        job_dict["status"] = FileJobStatus.FAILED.value
    else:
        job_dict["status"] = FileJobStatus.SUCCEEDED.value
    finally:
        logger.info(f"[{__name__}] indexing file({org}/{file_name}) end")
        job_dict["finished_at"] = math.floor(time.time())
        job_file.write_text(json.dumps(job_dict))
