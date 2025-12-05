import json
import os
import re
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel
from watchdog.observers import Observer

import interpreter as terps
from config import ReloadHandler, get_config
from formatting import markdown_to_html, remove_markdown_formatting
from images import get_mime_type, merge_image_and_json
from llm import rewrite_latest_message
from models import ChatRequest
from stores import close_stores, get_stores
from utils import (append_if_truthy, base64_to_dict, dict_to_base64, fnv1a_64,
                   natural_join, removeprefix_ci)

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


def get_variables_and_messages(messages: list[ChatMessage]) -> tuple[str, dict, list[ChatMessage], list[str], dict]:
    save_data = {}
    new_messages = []
    inputs = []
    game = None
    settings = {}
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
            settings = {
                "parser_augmentation": data_json.get("parser_aug", "disabled"),
                "output_augmentation": data_json.get("output_aug", "disabled"),
            }
            inputs_hash = data_json.get("inputs", None)
            if inputs_hash:
                inputs = stores.get("inputs").pop_when_ready(inputs_hash, timeout=5)
            else:
                inputs = None
            continue
        new_messages.append(message)
    return game, save_data, new_messages, inputs, settings


def parser_strip(input: str):
    return input.strip(" \t\n\r\f\v.!?")


rules_based_parser_remaps = {
    "take inventory": "inventory",
    "look around": "look",

    "head north": "north",
    "head northeast": "north",
    "head east": "east",
    "head southeast": "east",
    "head south": "south",
    "head southwest": "west",
    "head west": "west",
    "head northwest": "south",
    "head up": "up",
    "head down": "down",

    "head northwards": "north",
    "head northeastwards": "north",
    "head eastwards": "east",
    "head southeastwards": "east",
    "head southwards": "south",
    "head southwestwards": "west",
    "head westwards": "west",
    "head northwestwards": "south",
    "head upwards": "up",
    "head downwards": "down",
}


def get_last_nonsystem_message(messages: list):
    if not messages:
        return None
    last_message = None
    for message in messages:
        if message.role == MessageRole.USER or message.role == MessageRole.ASSISTANT:
            last_message = message
    return last_message


async def proxy_request(req: ChatRequest, request: Request):
    reqJson = await request.json()
    model_info = config.get_model(req.model)
    req_model = model_info["model"]
    req_base = model_info["base"]
    reqJson["model"] = req_model
    headers = {
        key: value for key, value in request.headers.items()
        if key.lower() not in {
            # RFC 2616 Hop-by-Hop
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            # Set by httpx
            "host",
            "content-length",
            # Set below
            "authorization",
        }
    }
    headers["Authorization"] = f"Bearer {config.get_key(model_info['keychain'])}"

    if reqJson.get("stream", False):
        async def event_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{req_base}/chat/completions",
                    headers=headers,
                    json=reqJson,
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{req_base}/chat/completions",
            headers=headers,
            json=reqJson,
        )
        return resp.json()


@app.post("/v1/chat/completions", tags=["Connection Wrapper"])
async def chat_completions(req: ChatRequest, request: Request):
    try:
        messages = req.messages
        last_message = get_last_nonsystem_message(messages)
        input = last_message.content.strip() if (last_message and last_message.role == MessageRole.USER) else ""
        game, save_data, messages, inputs, settings = get_variables_and_messages(messages)
        warnings = []

        if game is None:
            return await proxy_request(req, request)

        match settings["parser_augmentation"]:
            case "disabled":
                pass
            case "rules":
                input = remove_markdown_formatting(input)
                input = parser_strip(removeprefix_ci(input, "i "))
                input_lower = input.lower()
                for orig, remap in rules_based_parser_remaps.items():
                    if input_lower == orig:
                        input = remap
                        break
            case _:
                warnings.append(f"Setting \"parser_augmentation\" was set to \"{settings['parser_augmentation']}\", which is not yet implemented.")

        game_info = config.get_game(game)
        interpreter = config.get_terp(game_info["Interpreter"])
        terp_class = getattr(terps, interpreter["Class"], None)
        if not (terp_class and issubclass(terp_class, terps.Interpreter)):
            raise Exception(f"\"{interpreter['Class']}\" is not a valid interpreter.")

        save_data, input, output, variables = terp_class(interpreter["Path"], game_info["Path"], messages, save_data, interpreter.get("ExtraArgs", None))(input, inputs)
        thinking = None

        match settings["output_augmentation"]:
            case "disabled":
                pass
            case "rewrite":
                thinking = output
                output = await run_in_threadpool(rewrite_latest_message, output, messages, req.model, req.max_tokens, request)
            case _:
                warnings.append(f"Setting \"output_augmentation\" was set to \"{settings['output_augmentation']}\", which is not yet implemented.")

        ret_buffer = []
        if variables:
            ret_buffer.append(variables)
        if save_data:
            ret_buffer.append(f"<!--SAVE:\"{dict_to_base64(save_data)}\"-->")
        if input is not None:
            ret_buffer.append(f"<!--INPUT:\"{input}\"-->")
        for warning in warnings:
            ret_buffer.append(f"[Warning: {warning}]\n\n")
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
    if req.stream:
        async def stream_generator(result: str, thinking: str):
            chunks = []
            if thinking:
                chunks.append({
                    "choices": [{
                        "delta": {
                            "reasoning": thinking,
                            "reasoning_details": [{
                                "type": "reasoning.text",
                                "text": thinking,
                                "format": "unknown",
                                "index": 0
                            }]
                        }
                    }]
                })
            chunks.append({"choices": [{"delta": {"content": result}}]})
            chunks.append({"choices": [{"delta": {}}], "finish_reason": "stop"})
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(result, thinking), media_type="text/event-stream")
    else:  # Non-streaming response
        # TODO: Return reasoning chunk if applicable.
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
        description = game_info.get("Description", None)
        description = markdown_to_html(description) if description else "<p>No description provided.</p>"
        cover = game_info.get("Cover", "")
        game_list[game_id] = {
            "name": name,
            "author": get_author_line(author, authors),
            "description": description,
            "cover": cover,
        }
    return game_list


@app.get("/v1/vhelas/games/{game_id}/cover", tags=["Interactive Fiction Player"])
def get_game_cover(game_id: str):
    game_info = config.get_game(game_id)
    cover = game_info.get("Cover", "")
    if not cover:
        # TODO: If there isn't a cover, generate a placeholder.
        raise HTTPException(status_code=404, detail="Cover not available.")
    if os.path.exists(cover):
        return FileResponse(cover, media_type=get_mime_type(cover))


def get_game_character_card_json(game_id: str):
    game_info = config.get_game(game_id)
    name = game_info.get("ShortName", game_info.get("Name", game_id)).replace("/", "\u2215").replace(":", "\uA789")
    author = game_info.get("Author", None)
    authors = game_info.get("Authors", [])
    description = game_info.get("Description", None)
    description = markdown_to_html(description) if description else "<p>No description provided.</p>"
    description = f"<h1>{game_info.get('Name', game_id)}</h1>{description}"
    tags = ["Interactive Fiction"]
    append_if_truthy(tags, game_info.get("Genre", None))
    difficulty = game_info.get("Difficulty", None)
    if difficulty:
        append_if_truthy(tags, f"{difficulty} Difficulty")

    interpreter = config.get_terp(game_info["Interpreter"])
    terp_class = getattr(terps, interpreter["Class"], None)
    if not (terp_class and issubclass(terp_class, terps.Interpreter)):
        raise Exception(f"\"{interpreter['Class']}\" is not a valid interpreter.")

    save_data, _, output, variables = terp_class(interpreter["Path"], game_info["Path"], [], None, interpreter.get("ExtraArgs", None))(None, None)

    ret_buffer = [f"<!--GAME:\"{game_id}\"-->"]
    append_if_truthy(ret_buffer, variables)
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
            "mes_example": "",
            "scenario": "",
            "creator_notes": description,
            "system_prompt": "",
            "post_history_instructions": "",
            "alternate_greetings": [],
            "tags": tags,
            "creator": get_author_line(author, authors),
            "character_version": "main",
            "extensions": {
                "vhelas_game": game_id
            },
            "character_book": None,
        }
    }


@app.get("/v1/vhelas/games/{game_id}", tags=["Interactive Fiction Player"])
def get_game_character_card(game_id: str):
    game_info = config.get_game(game_id)
    portrait = game_info.get("Portrait", game_info.get("Cover", None))
    if not portrait:
        return get_game_character_card_json(game_id)
    return StreamingResponse(merge_image_and_json(portrait, json.dumps(get_game_character_card_json(game_id))), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8786)
