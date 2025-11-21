from llama_index.core.llms import ChatMessage
from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 512
    stream: bool | None = False
    tools: list[dict] | None = None
    tool_choice: str | None = None
