from fastapi import FastAPI

from app.api.routes import documents, health, query
from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
