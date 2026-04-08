# main.py
from fastapi import FastAPI, HTTPException, Query
from uuid import uuid4
from datetime import datetime
import json
from typing import List
from models import ChatRequest, ChatResponse, ChatUpdate, ChatListItem
from tools import run_langchain_query_tool
from langagent import build_schema_cache, load_schema_cache

app = FastAPI(title="ColdLion Chat API")

# Simulate in-memory chat storage for demo
chat_store = {}

@app.post("/prompt", response_model=ChatResponse)
def create_prompt(request: ChatRequest):
    """Handle user question and generate chat response"""
    try:
        chat_id = str(uuid4())
        chat_title = request.user_prompt[:50]
        ai_response = run_langchain_query_tool(request.user_prompt)
        # Handle str, dict, or list outputs
        if isinstance(ai_response, list):
            if len(ai_response) > 0:
                first_item = ai_response[0]
                if isinstance(first_item, dict):
                    response_text = first_item.get("conversational_response") or first_item.get("result") or str(first_item)
                else:
                    response_text = str(first_item)
            else:
                response_text = "No response generated"
        elif isinstance(ai_response, dict):
            response_text = ai_response.get("result") or str(ai_response)
        else:
            response_text = str(ai_response)
        chat_data = ChatResponse(
            chat_id=chat_id,
            chat_title=chat_title,
            response_text=response_text,
            created_at=datetime.now()
        )
        chat_store[chat_id] = chat_data.dict()
        return chat_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chats", response_model=List[ChatListItem])
def list_chats(search_query: str = Query("", description="Search chat titles")):
    """Fetch chat history with optional search"""
    chats = [ChatListItem(**c) for c in chat_store.values()]
    if search_query:
        chats = [c for c in chats if search_query.lower() in c.chat_title.lower()]
    return chats


@app.get("/chats/{chat_id}", response_model=ChatResponse)
def get_chat(chat_id: str):
    """Fetch single chat by ID"""
    chat = chat_store.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.put("/chats/{chat_id}", response_model=ChatResponse)
def update_chat(chat_id: str, update: ChatUpdate):
    """Update chat title or response"""
    chat = chat_store.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if update.chat_title:
        chat["chat_title"] = update.chat_title
    if update.response_text:
        chat["response_text"] = update.response_text
    chat_store[chat_id] = chat
    return chat


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    """Delete chat"""
    if chat_id in chat_store:
        del chat_store[chat_id]
        return {"message": "Chat deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Chat not found")


@app.get("/refresh_tables")
def refresh_schema():
    """Rebuild the schema cache"""
    schema = build_schema_cache()
    return {"message": "Schema cache rebuilt", "tables_cached": len(schema)}
