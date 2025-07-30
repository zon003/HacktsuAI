from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # "user" または "assistant"
    content: str

class ChatRequest(BaseModel):
    userId: str
    message: str
    chatHistory: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str