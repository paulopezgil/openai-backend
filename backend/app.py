from fastapi import FastAPI


app = FastAPI(title="OpenAI Compatible API", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "OpenAI Compatible API"}