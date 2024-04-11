import asyncio
import json
import math
import time
from typing import Annotated, List
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Header,
    Path,
    BackgroundTasks,
)
from llama_index.core import Document, SimpleDirectoryReader
from ..models.file import (
    CreateFileJonRequest,
    FileDocument,
    SearchFilesRequest,
    SearchResult,
    FileJobStatus,
    FileJob,
)
from ..dependencies import (
    authenticate,
    get_file_path,
    get_index_name,
    get_job_file_path,
    get_org_header,
    parse_mime_type,
)
from ..tasks.index import index_file
from ..providers.vector_store.factory import get_vector_store, get_vector_store_index

router = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(authenticate), Depends(get_org_header)],
    responses={
        401: {
            "type": "about:blank",
            "status": 401,
            "title": "Unauthorized",
        },
        404: {
            "type": "about:blank",
            "status": 404,
            "title": "Not Found",
        },
    },
)


def _convert_to_file_document(doc: Document):
    return FileDocument(
        id=doc.get_doc_id(),
        text=doc.get_text(),
    )


@router.post("/{file_id}/job")
async def create_file_job(
    x_retrieval_organization: Annotated[str, Header()],
    file_id: Annotated[str, Path(description="The ID of file to get documents")],
    req: CreateFileJonRequest,
    tasks: BackgroundTasks,
):
    file_path = get_file_path(x_retrieval_organization, req.file_name)
    if not file_path.exists():
        raise HTTPException(status_code=404)

    job_file_path = get_job_file_path(x_retrieval_organization, file_id)
    if job_file_path.exists():
        raise HTTPException(status_code=409)

    status = FileJobStatus.QUEUED.value
    job = FileJob(status=status, created_at=math.floor(time.time()))

    job_file_path.parent.mkdir(parents=True, exist_ok=True)
    job_file_path.touch()
    job_file_path.write_text(job.model_dump_json())

    file_type = parse_mime_type(req.file_name)
    file_type = req.file_type if file_type is None else file_type
    if not file_type:
        file_type = req.file_type
    tasks.add_task(
        index_file,
        x_retrieval_organization,
        file_id,
        req.file_name,
        file_type,
    )
    return job


@router.get("/{file_id}/job")
async def get_file_job(
    x_retrieval_organization: Annotated[str, Header()],
    file_id: Annotated[str, Path(description="The ID of file to get documents")],
):
    file_path = get_job_file_path(x_retrieval_organization, file_id)
    if not file_path.exists():
        raise HTTPException(status_code=404)

    return json.loads(file_path.read_text())


@router.get("/{file_name}/documents")
async def get_file_documents(
    x_retrieval_organization: Annotated[str, Header()],
    file_name: Annotated[str, Path(description="The ID of file to get documents")],
) -> List[FileDocument]:
    file_path = get_file_path(x_retrieval_organization, file_name)
    if not file_path.exists():
        raise HTTPException(status_code=404)

    reader = SimpleDirectoryReader(input_files=[str(file_path)])
    documents = reader.load_data()
    return list(map(_convert_to_file_document, documents))


@router.post("/search")
async def search_files(
    x_retrieval_organization: Annotated[str, Header()], req: SearchFilesRequest
) -> List[SearchResult]:
    # file name -> index name
    index_names = list(
        map(lambda id: get_index_name(x_retrieval_organization, id), req.file_ids)
    )
    # index name -> vector store index
    indexes = list(map(get_vector_store_index, index_names))
    # index -> retriever
    retrievers = list(map(lambda idx: idx.as_retriever(), indexes))
    # each retriever retreves input
    task_list = list(map(lambda r: r.aretrieve(req.query), retrievers))
    nodes_list = await asyncio.gather(*task_list)
    # flatten nodes list
    nodes = [node for nodes in nodes_list for node in nodes]

    return list(
        map(
            lambda node: SearchResult(
                text=node.text, metadata=node.metadata, score=node.score
            ),
            nodes,
        )
    )
