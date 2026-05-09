from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import (
    chat_router,
    models_router,
    embeddings_router,
    completions_router,
    responses_router,
    files_router,
    moderations_router,
    health_router,
    debug_router,
)

app = FastAPI(title="OpenAI Compatible API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(debug_router)
app.include_router(embeddings_router)
app.include_router(completions_router)
app.include_router(responses_router)
app.include_router(files_router)
app.include_router(moderations_router)


@app.get("/")
async def root():
    return {"message": "OpenAI Compatible API"}