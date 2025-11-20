
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from watchdog.observers import Observer

from config import ReloadHandler, get_config

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    observer = Observer()
    observer.schedule(ReloadHandler(), ".", recursive=False)
    observer.start()
    print("Observer started")
    yield
    observer.stop()
    # This blocks until we're done stopping.
    observer.join()
    print("Observer stopped")

app = FastAPI(
    lifespan=lifespan,
    title="Vhelas",
    description="OpenAI-compatible API wrapper for playing interactive fiction.",
    version="0.1.0"
)


@app.get("/v1/models", tags=["Connection Wrapper"])
async def list_models(request: Request):
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": model_info["owner"],
                "permission": [],
            }
            for model_id, model_info in config.get_models()
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8786)
