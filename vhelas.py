from contextlib import asynccontextmanager

import json
import re
import uuid
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from watchdog.observers import Observer
from llama_index.core.base.llms.types import MessageRole

from config import ReloadHandler, get_config
from interpreter import RemGlkGlulxeInterpreter
from models import ChatMessage, ChatRequest
from stores import get_stores, close_stores
from utils import base64_to_dict, dict_to_base64, fnv1a_64

config = get_config()
stores = get_stores()


@asynccontextmanager
async def lifespan(app: FastAPI):
    observer = Observer()
    observer.schedule(ReloadHandler(), ".", recursive=False)
    observer.start()
    yield
    observer.stop()
    observer.join()  # This blocks until we're done stopping.
    print("Configuration observer stopped.")
    if close_stores():
        print("Persistent data stores closed.")

app = FastAPI(
    lifespan=lifespan,
    title="Vhelas",
    description="OpenAI-compatible API wrapper for playing interactive fiction.",
    version="0.1.0"
)


@app.get("/v1/models", tags=["Connection Wrapper"])
def list_models(request: Request):
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


def get_variables_and_messages(messages: list[ChatMessage]) -> tuple[dict, list[ChatMessage]]:
    save_data = {}
    new_messages = []
    inputs = []
    for message in messages:
        content = message.content
        if not content:
            continue
        save_match = re.search(r"<!--SAVE:(.*?)-->", content, re.DOTALL)
        if save_match:
            captured = save_match.group(1)
            as_json = json.loads(captured)
            save_data = base64_to_dict(stores.get("saves").pop_when_ready(as_json, timeout=5))
            continue

        def input_replacer(match):
            captured = match.group(1)
            as_json = json.loads(captured)
            if isinstance(as_json, list):
                inputs.extend(as_json)
            else:
                inputs.append(as_json)
            return ""
        stripped_content = re.sub(r"<!--INPUT:(.*?)-->", input_replacer, content, flags=re.DOTALL).strip()
        if stripped_content:
            new_messages.append(message)
    return save_data, new_messages, inputs


@app.post("/v1/chat/completions", tags=["Connection Wrapper"])
def chat_completions(request: ChatRequest):
    try:
        messages = request.messages
        last_message = messages[-1] if messages else {}
        input = last_message.content if last_message.role == MessageRole.USER else ""
        save_data, messages, inputs = get_variables_and_messages(messages)

        save_data, input, output = RemGlkGlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", messages, save_data)(input, inputs)

        result = f"<!--SAVE:\"{dict_to_base64(save_data)}\"--><!--INPUT:\"{input}\"-->{output}"
    except Exception as e:
        return JSONResponse(content={
            "error": {
                "message": repr(e),
                "type": "server_error",
                "param": None,
                "code": "500"
            }
        }, status_code=500)
    if request.stream:
        async def stream_generator(result: str):
            for chunk in [
                {"choices": [{"delta": {"content": result}}]},
                {"choices": [{"delta": {}}], "finish_reason": "stop"}
            ]:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(result), media_type="text/event-stream")
    else:  # Non-streaming response
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


class SavePackage(BaseModel):
    save_data: str
    hash: str


@app.post("/v1/save", tags=["Interactive Fiction Player"])
def post_save(package: SavePackage):
    try:
        save_data = package.save_data
        reported_fnv1a_hash = package.hash
        calculated_fnv1a_hash = str(fnv1a_64(save_data))
        if reported_fnv1a_hash == calculated_fnv1a_hash:
            stores.get("saves").set(calculated_fnv1a_hash, save_data)
        else:
            print(f"Reported hash of {reported_fnv1a_hash} doesn't match actual hash of {calculated_fnv1a_hash}. Rejecting save file.")
    except Exception as e:
        return JSONResponse(content={
            "error": {
                "message": repr(e),
                "type": "server_error",
                "param": None,
                "code": "500"
            }
        }, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8786)
