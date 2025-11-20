from fastapi import Request
from llama_index.llms.openai_like import OpenAILike

from config import get_config
from exceptions import MissingLLMModelException


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
    if request:
        if request.headers:
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
