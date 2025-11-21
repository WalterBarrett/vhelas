
from contextlib import asynccontextmanager

import json
import re
import uuid
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from watchdog.observers import Observer

from config import ReloadHandler, get_config
from interpreter import base64_to_dict, dict_to_base64, GlulxeInterpreter

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


def get_variables(messages: list[str]):
    save_data = {}
    for message in messages:
        content = message.get("content", "")
        if content:
            save_regex = r"<!--SAVE:(.*?)-->"
            match = re.search(save_regex, content, re.DOTALL)
            if match:
                captured = match.group(1)
                save_data = base64_to_dict(captured)
                new_text = re.sub(save_regex, "", content, flags=re.DOTALL)
    return save_data


def get_output(remglk: dict):
    output_buffer = []
    windows = remglk.get("windows", [])
    window_ids = set()
    for window in windows:
        window_type = window.get("type", None)
        if window_type == "buffer" or window_type == 3:
            window_ids.add(window.get("id", window.get("tag", None)))
    content = remglk.get("content", [])
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
                            output_buffer.append(text)
                    output_buffer.append("\n")

    print(json.dumps(remglk))
    return "".join(output_buffer)


@app.post("/v1/chat/completions", tags=["Connection Wrapper"])
async def chat_completions(request: Request):
    body = await request.json()
    try:
        messages = body.get("messages", [])
        last_message = messages[-1] if messages else {}
        if last_message.get("role", None) == "user":
            input = last_message.get("content", "")
        else:
            input = ""
        save_data = get_variables(messages)

        glulxe = GlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", save_data)
        save_data = glulxe.run(input)
        remglk = save_data.pop("remglk")
        autosave_json = save_data.get("autosave.json", {})
        autosave_json_windows = autosave_json.get("windows", {})
        if autosave_json_windows:
            text = get_output({"windows": autosave_json_windows} | remglk)
        else:
            text = get_output(remglk)

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
