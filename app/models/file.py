from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict


class FileDocument(BaseModel):
    id: str = Field(description="The ID of document.")
    text: str = Field(description="The text of document.")


class SearchFilesRequest(BaseModel):
    file_ids: list[str] = Field(
        description="The ID of files which which are to search.",
        min_length=1,
        max_length=10,
    )
    query: str = Field(description="The query of search.")


class SearchResult(BaseModel):
    text: str = Field(description="The relevant text given input.")
    metadata: Dict[str, Any] = Field(
        description="A dictionary of annotations that can be appended to the text."
    )
    score: float | None = Field(
        default=None,
        description="This is a measure of similarity between this text and the input. The higher the score, the more they are similar.",
    )


class FileJobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FileJob(BaseModel):
    # file_name: str = Field(description="The file name.")
    status: str = Field(description="The status of file job.")
    tokens: int | None = None
    created_at: int = Field(description="")
    finished_at: int | None = Field(default=None, description="")


class CreateFileJonRequest(BaseModel):
    file_name: str = Field(description="The name of file.")
    file_type: str = Field(description="The mime type of file.")
