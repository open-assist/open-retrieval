from fastapi import FastAPI
from .routers import files
from .providers.embedding.factory import set_embed_model


set_embed_model()

app = FastAPI(
    title="OpenRetrieval", description="The OpenAPI of OpenRetrieval", openapi_url="/"
)
app.include_router(files.router)
