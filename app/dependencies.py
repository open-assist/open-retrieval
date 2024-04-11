import logging
import os
from pathlib import Path
from ulid import ULID
from typing import Annotated
from fastapi import Header, HTTPException


TOKEN = os.environ.get("RETRIEVAL_TOKEN", str(ULID()))
DATA_DIR = os.environ.get("DATA_DIR", "/tmp")


logger = logging.getLogger("uvicorn")


async def authenticate(
    x_retrieval_token: Annotated[
        str,
        Header(
            description="To authenticate your request, you will need to provide an authentication token."
        ),
    ]
):
    if x_retrieval_token != TOKEN:
        raise HTTPException(status_code=401, detail="X-Retrieval-Token header invalid.")


async def get_org_header(
    x_retrieval_organization: Annotated[
        str,
        Header(description="Specify which organization is used for an API request."),
    ]
):
    if not x_retrieval_organization:
        raise HTTPException(
            status_code=400, detail="Need X-Retrieval-Organization header."
        )


def get_index_name(org: str, file_id: str):
    return f"{org}-{file_id.lower()}"


def get_file_path(org: str, file_id: str):
    """
    Generate a file path for a file in the organization's data directory.

    Args:
        org (str): The organization identifier.
        file_id (str): The unique identifier for the file.

    Returns:
        Path: The file path for the file in the organization's data directory.
    """
    return Path(f"{DATA_DIR}/{org}/{file_id}")


def get_job_file_path(org: str, file_name: str):
    """
    Generate a file path for a job file in the organization's index jobs directory.

    Args:
        org (str): The organization identifier.
        job_id (str): The unique identifier for the job.

    Returns:
        Path: The file path for the job file.
    """
    return Path(f"{DATA_DIR}/{org}-file-jobs/{file_name}")


def parse_mime_type(file_name: str):
    """
    Parse the MIME type of a file based on its file extension.

    Args:
        file_name (str): The name of the file.

    Returns:
        str: The appropriate MIME type string if recognized, otherwise None.
    """
    suffix = file_name.split(".")[-1]
    match suffix:
        case "cpp":
            return "text/x-c++"
        case "cs":
            return "text/x-csharp"
        case "py":
            return "text/x-python"
        case "rb":
            return "text/x-ruby"
        case _:
            return None
