from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    userId: str
    jwtToken: str
    chatHistory: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str