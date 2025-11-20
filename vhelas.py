
from contextlib import asynccontextmanager

import json
import uuid
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
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


@app.post("/v1/chat/completions", tags=["Connection Wrapper"])
async def chat_completions(request: Request):
    body = await request.json()
    try:
        result = "Hello world!"
    except Exception as e:
        return JSONResponse(content={
            "error": {
                "message": repr(e),
                "type": "server_error",
                "param": None,
                "code": "500"
            }
        }, status_code=500)
    if body.get("stream", False):
        async def stream_generator(result: str):
            for chunk in [
                {"choices": [{"delta": {"content": result}}]},
                {"choices": [{"delta": {}}], "finish_reason": "stop"}
            ]:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(result), media_type="text/event-stream")
    else: # Non-streaming response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop"
                }
            ]
        }
        return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8786)
