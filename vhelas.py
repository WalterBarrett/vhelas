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

import interpreter as terps
from config import ReloadHandler, get_config
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
    game = None
    gamestart = None
    for message in messages:
        content = message.content
        print(f"content: {content}")
        if not content:
            continue
        save_match = re.search(r"<!--SAVE:(.*?)-->", content, re.DOTALL)
        if save_match:
            captured = save_match.group(1)
            as_json = json.loads(captured)
            save_data = base64_to_dict(stores.get("saves").pop_when_ready(as_json, timeout=5))
            continue

        def input_replacer(match):
            as_json = json.loads(match.group(1))
            if isinstance(as_json, list):
                inputs.extend(as_json)
            else:
                inputs.append(as_json)
            return ""

        def game_replacer(match):
            nonlocal game
            game = json.loads(match.group(1))
            print(f"GAME: {game}")
            return ""

        def gamestart_replacer(match):
            nonlocal gamestart
            gamestart = json.loads(match.group(1))
            print(f"GAMESTART: {gamestart}")
            return ""

        content = re.sub(r"<!--GAMESTART:(.*?)-->", gamestart_replacer, content, flags=re.DOTALL).strip()
        content = re.sub(r"<!--GAME:(.*?)-->", game_replacer, content, flags=re.DOTALL).strip()
        content = re.sub(r"<!--INPUT:(.*?)-->", input_replacer, content, flags=re.DOTALL).strip()
        if content:
            new_messages.append(message)
    if not gamestart:
        inputs = None
    return game, save_data, new_messages, inputs


@app.post("/v1/chat/completions", tags=["Connection Wrapper"])
def chat_completions(request: ChatRequest):
    try:
        messages = request.messages
        last_message = messages[-1] if messages else {}
        input = last_message.content if last_message.role == MessageRole.USER else ""
        game, save_data, messages, inputs = get_variables_and_messages(messages)

        if game is None:
            raise Exception("No game is set.")

        game_info = config.get_game(game)
        print(game_info)
        interpreter = config.get_terp(game_info["Interpreter"])
        print(interpreter)
        terp_class = getattr(terps, interpreter["Class"], None)
        print(terp_class)
        if not (terp_class and issubclass(terp_class, terps.Interpreter)):
            raise Exception(f"\"{interpreter['Class']}\" is not a valid interpreter.")

        save_data, input, output = terp_class(interpreter["Path"], game_info["Path"], messages, save_data)(input, inputs)

        ret_buffer = []
        if save_data:
            ret_buffer.append(f"<!--SAVE:\"{dict_to_base64(save_data)}\"-->")
        if input is not None:
            ret_buffer.append(f"<!--INPUT:\"{input}\"-->")
        ret_buffer.append(output)
        result = "".join(ret_buffer)
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
            print("RESULT FOR SILLYTAVERN: ", result)
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
