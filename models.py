# models.py
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str      # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    userId: str
    message: str
    jwtToken: Optional[str] = None             # ← ここを Optional & デフォルト None に
    chatHistory: List[ChatMessage] = []        # デフォルト空リスト

class ChatResponse(BaseModel):
    response: str
