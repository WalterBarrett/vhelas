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
from interpreter import GlulxeInterpreter
from models import ChatRequest
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


def get_variables(messages: list[str]):
    save_data = {}
    for message in messages:
        content = message.content
        if content:
            save_regex = r"<!--SAVE:(.*?)-->"
            match = re.search(save_regex, content, re.DOTALL)
            if match:
                captured = match.group(1)
                save_data = base64_to_dict(stores.get("saves").pop_when_ready(captured, timeout=5))
                new_text = re.sub(save_regex, "", content, flags=re.DOTALL)
    return save_data


def get_output(remglk: list[dict], input: str | None, fallback_windows: list[dict] = None):
    output_buffer = []
    for rgo in remglk:
        type = rgo.get("type", None)
        if type == "update":
            windows = rgo.get("windows", fallback_windows)
            window_ids = set()
            for window in windows:
                window_type = window.get("type", None)
                if window_type == "buffer" or window_type == 3:
                    window_ids.add(window.get("id", window.get("tag", None)))
            content = rgo.get("content", [])
            if content:
                for window in content:
                    window_id = window.get("id", None)
                    if window_id in window_ids:
                        for text in window.get("text", None):
                            inner_content = text.get("content", {})
                            if inner_content:
                                for inner in inner_content:
                                    style = inner.get("style", None)
                                    text = inner.get("text", "")
                                    match style:
                                        case "emphasized":
                                            output_buffer.append(f"*{text}*")
                                        case "preformatted":
                                            output_buffer.append(f"`{text}`")
                                        case "header":
                                            output_buffer.append(f"# {text}")
                                        case "subheader":
                                            output_buffer.append(f"## {text}")
                                        case "alert":
                                            output_buffer.append(f"**{text}**")
                                        case "note":
                                            output_buffer.append(f"*{text}*")
                                        case "blockquote":
                                            output_buffer.append(f"> {text}")
                                        case "input":
                                            if text.strip() != input.strip():
                                                output_buffer.append(f"**`{text}`**")
                                        case "user1":
                                            output_buffer.append(f"*{text}*")
                                        case "user2":
                                            output_buffer.append(f"*{text}*")
                                        case "normal" | "unknown" | _:
                                            output_buffer.append(text)
                            output_buffer.append("\n")
        elif type == "error":
            output_buffer.append(f"[Error: {rgo.get('message', 'Unspecified error.')}]\n")

    # print(json.dumps(remglk))
    return "".join(output_buffer).rstrip("> \t\n\r")


@app.post("/v1/chat/completions", tags=["Connection Wrapper"])
def chat_completions(request: ChatRequest):
    try:
        messages = request.messages
        last_message = messages[-1] if messages else {}
        input = last_message.content if last_message.role == MessageRole.USER else ""
        save_data = get_variables(messages)

        glulxe = GlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", save_data)
        save_data, input = glulxe.run(input)
        remglk = save_data.pop("remglk")
        autosave_json = save_data.get("autosave.json", {})
        text = get_output(remglk, input, autosave_json.get("windows", {}))

        result = f"<!--SAVE:\"{dict_to_base64(save_data)}\"-->{text}"
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
