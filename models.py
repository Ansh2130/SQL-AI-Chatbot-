# models.py

from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict
from datetime import datetime

# 🧠 Model for incoming chat question
class ChatRequest(BaseModel):
    user_prompt: str = Field(..., example="Show top customers for 2024")


# 🧩 Main chat response model (returned to frontend)
class ChatResponse(BaseModel):
    chat_id: str = Field(..., example="8f23b5d6-1234-45de-8901-abcdef987654")
    chat_title: str = Field(..., example="Show top customers for 2024")
    response_text: Union[str, Dict] = Field(
        ..., 
        example="The top 5 customers are XYZ Ltd, ABC Corp..."
    )
    chart_image: Optional[str] = Field(
        None,
        description="Base64 image of chart (if available)"
    )
    query_text: Optional[str] = Field(
        None,
        description="Generated SQL query (if available)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when chat was created"
    )


# 🧱 For updating an existing chat (PUT)
class ChatUpdate(BaseModel):
    chat_title: Optional[str] = Field(None, example="Updated title")
    response_text: Optional[str] = Field(None, example="Updated response text")


# 📜 For chat listing view (GET /chats)
class ChatListItem(BaseModel):
    chat_id: str
    chat_title: str
    created_at: datetime
