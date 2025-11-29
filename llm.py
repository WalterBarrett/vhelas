import json

from fastapi import Request
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.llms.openai_like import OpenAILike

from config import get_config
from cooldown import truthy_validator, use_cooldown
from exceptions import MissingLLMModelException, NoLLMOutputException
from models import JsonChatResponse


def apply_stops(text: str, stops: list[str]) -> str:
    """Truncate text at the first occurrence of any stop string."""
    min_index = None
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            if min_index is None or idx < min_index:
                min_index = idx
    return text if min_index is None else text[:min_index]


def load_llm_model(model: str, request: Request, json=False):
    config = get_config()
    additional_kwargs = {
        "stop": config.get_setting("StopStrings", []),
    }
    if json:
        additional_kwargs["response_format"] = {
            "type": "json_object"
        }

    http_referer = config.get_setting("WrapperDisplayURI", None)
    x_title = config.get_setting("WrapperDisplayTitle", None)
    if request and request.headers:
        http_referer = request.headers.get("HTTP-Referer", http_referer)
        x_title = request.headers.get("X-Title", x_title)
    default_headers = {}
    if http_referer:
        default_headers["HTTP-Referer"] = http_referer
    if x_title:
        default_headers["X-Title"] = x_title

    model_info = config.get_model(model)
    if not model_info:
        raise MissingLLMModelException()
    return OpenAILike(
        model=model_info["model"],
        api_key=config.get_key(model_info["keychain"]),
        api_base=model_info["base"],
        is_chat_model=True,
        additional_kwargs=additional_kwargs,
        default_headers=default_headers
    )


@use_cooldown("llm_pool", validator=truthy_validator)
def rewrite_latest_message(latest_message: str, messages: list[ChatMessage], model: str, max_tokens: int, request: Request) -> str:
    config = get_config()
    llm = load_llm_model(model, request, json=True)
    prompt = f"You are a script rewriter. Rewrite the following response according to your instructions. Do not explain, justify, or add commentary about your rewrites. Output your reponse in JSON format (a single dict with a \"response\" key).\n\nText: {latest_message}"

    new_messages = [ChatMessage(role=MessageRole.SYSTEM, content=prompt)] + messages
    chat_response: ChatResponse = llm.chat(new_messages, max_tokens=max_tokens)
    response = apply_stops(chat_response.message.content, config.get_setting("StopStrings", [])).strip().removeprefix("```json").strip()
    if not response:
        raise NoLLMOutputException()
    try:
        decoded, _ = json.JSONDecoder().raw_decode(response)
        result = JsonChatResponse(**decoded)
    except Exception as e:
        print(f"{e}: \"\"\"{response}\"\"\"")
        raise
    return result.response
