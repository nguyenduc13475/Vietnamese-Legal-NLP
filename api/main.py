from fastapi import FastAPI

import api.config  # noqa: F401 (Imported first to initialize env and warnings)
from api.routes import router

app = FastAPI(
    title="Vietnamese Legal NLP API",
    description="API for Legal Contract Parsing, Semantic Analysis, and Q&A System",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def read_root():
    return {"message": "Legal NLP API is running. Visit /docs for Swagger UI."}
