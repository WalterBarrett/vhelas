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
from utils import base64_to_dict, dict_to_base64, fnv1a_64, natural_join

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
    for message in messages:
        content = message.content
        if not content:
            continue
        data_match = re.search(r"<!--DATA:(.*?)-->", content, re.DOTALL)
        if data_match:
            captured = data_match.group(1)
            data_json = json.loads(captured)
            save_hash = data_json.get("save", None)
            if save_hash:
                save_data = base64_to_dict(stores.get("saves").pop_when_ready(save_hash, timeout=5))
            else:
                save_data = None
            game = data_json.get("game", None)
            inputs_hash = data_json.get("inputs", None)
            if inputs_hash:
                inputs = stores.get("inputs").pop_when_ready(inputs_hash, timeout=5)
            else:
                inputs = None
            continue
        new_messages.append(message)
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
        interpreter = config.get_terp(game_info["Interpreter"])
        terp_class = getattr(terps, interpreter["Class"], None)
        if not (terp_class and issubclass(terp_class, terps.Interpreter)):
            raise Exception(f"\"{interpreter['Class']}\" is not a valid interpreter.")

        save_data, input, output = terp_class(interpreter["Path"], game_info["Path"], messages, save_data, interpreter.get("ExtraArgs", None))(input, inputs)

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


class StorePackage(BaseModel):
    data: str
    hash: str


@app.post("/v1/vhelas/{store}", tags=["Interactive Fiction Player"])
def post_vhelas_store(store: str, package: StorePackage):
    try:
        match store:
            case "save" | "input":
                pass
            case _:
                raise Exception(f"Invalid store \"{store}\".")
        data = package.data
        reported_fnv1a_hash = package.hash
        calculated_fnv1a_hash = str(fnv1a_64(data))
        if reported_fnv1a_hash == calculated_fnv1a_hash:
            if store != "save":
                data = json.loads(data)
            stores.get(f"{store}s").set(calculated_fnv1a_hash, data)
        else:
            raise Exception(f"Reported hash of {reported_fnv1a_hash} doesn't match actual hash of {calculated_fnv1a_hash}. Rejecting {store} file.")
    except Exception as e:
        return JSONResponse(content={
            "error": {
                "message": repr(e),
                "type": "server_error",
                "param": None,
                "code": "500"
            }
        }, status_code=500)


def get_author_line(author: str | None, authors: list[str] | None) -> str:
    if not author and not authors:
        return "Not Specified"
    if not author and authors:
        return natural_join(authors)
    if author and authors:
        return f"{author} ({natural_join(authors)})"
    return author


@app.get("/v1/vhelas/games", tags=["Interactive Fiction Player"])
def list_games():
    game_list = {}
    for game_id, game_info in config.get_games():
        name = game_info.get("Name", game_id)
        author = game_info.get("Author", None)
        authors = game_info.get("Authors", [])
        description = game_info.get("Description", "No description provided.")
        cover = game_info.get("Cover", "")
        game_list[game_id] = {

            "name": name,
            "author": get_author_line(author, authors),
            "description": description,
            "cover": cover,
        }
    return game_list


@app.get("/v1/vhelas/games/{game_id}", tags=["Interactive Fiction Player"])
def get_game_character_card(game_id: str):
    game_info = config.get_game(game_id)
    name = game_info.get("Name", game_id).replace("/", "\u2215").replace(":", "\uA789")
    author = game_info.get("Author", None)
    authors = game_info.get("Authors", [])
    description = game_info.get("Description", "No description provided.")
    cover = game_info.get("Cover", "")

    interpreter = config.get_terp(game_info["Interpreter"])
    terp_class = getattr(terps, interpreter["Class"], None)
    if not (terp_class and issubclass(terp_class, terps.Interpreter)):
        raise Exception(f"\"{interpreter['Class']}\" is not a valid interpreter.")

    save_data, _, output = terp_class(interpreter["Path"], game_info["Path"], [], None, interpreter.get("ExtraArgs", None))(None, None)

    ret_buffer = [f"<!--GAME:\"{game_id}\"-->"]
    if save_data:
        ret_buffer.append(f"<!--SAVE:\"{dict_to_base64(save_data)}\"-->")
    ret_buffer.append(output)

    first_mes = "".join(ret_buffer)

    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": "",
            "personality": "",
            "first_mes": first_mes,
            "avatar": cover,
            "mes_example": "",
            "scenario": "",
            "creator_notes": description,
            "system_prompt": "",
            "post_history_instructions": "",
            "alternate_greetings": [],
            "tags": [
                "Interactive Fiction",
            ],
            "creator": get_author_line(author, authors),
            "character_version": "main",
            "extensions": {
                "vhelas_game": game_id
            },
            "character_book": None,
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8786)
