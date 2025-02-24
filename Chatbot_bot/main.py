from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Groq Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("static", StaticFiles(directory="static"), name="static")

# Templates configuration
templates = Jinja2Templates(directory="templates")


# Models for request/response
class Message(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    messages: List[Message]


class ChatRequest(BaseModel):
    message: str
    persona: str = "Default"
    model: str = "llama3-70b-8192"
    memory_length: int = 5


class ChatResponse(BaseModel):
    response: str
    chat_history: List[Message]


# Store chat sessions (in a real application, use a proper database)
chat_sessions = {}


def get_custom_prompt(persona: str) -> PromptTemplate:
    """Get custom prompt template based on selected persona."""
    personas = {
        "Default": """You are a friendly and helpful AI assistant, providing clear, concise, and accurate responses.
                     Focus on being approachable and ensuring the user feels understood and supported.
                     Current conversation:
                     {history}
                     Human: {input}
                     AI:""",

        "Expert": """You are a highly knowledgeable and authoritative expert across various fields.
                    Offer in-depth, precise, and technical explanations, citing examples or relevant research when necessary.
                    Avoid jargon when possible, but feel free to introduce advanced concepts where appropriate.
                    Current conversation:
                    {history}
                    Human: {input}
                    Expert:""",

        "Creative": """You are an imaginative and inventive AI with a flair for creative problem-solving and thinking outside the box.
                      Use metaphors, vivid descriptions, and unconventional ideas to inspire and captivate the user.
                      Feel free to suggest unique approaches or surprising solutions to problems.
                      Current conversation:
                      {history}
                      Human: {input}
                      Creative AI:"""
    }

    return PromptTemplate(
        input_variables=["history", "input"],
        template=personas.get(persona, personas["Default"])
    )


def get_chat_chain(model: str, memory_length: int, persona: str):
    """Initialize chat chain with specified parameters."""
    try:
        # Initialize Groq chat
        groq_chat = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model
        )

        # Set up memory
        memory = ConversationBufferWindowMemory(k=memory_length)

        # Create conversation chain
        conversation = ConversationChain(
            llm=groq_chat,
            memory=memory,
            prompt=get_custom_prompt(persona)
        )

        return conversation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize chat: {str(e)}")


@app.get("/")
async def home(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests."""
    try:
        # Create or get conversation chain
        conversation = get_chat_chain(
            model=request.model,
            memory_length=request.memory_length,
            persona=request.persona
        )

        # Get response from conversation chain
        response = conversation(request.message)

        # Prepare chat history
        chat_history = [
            Message(role="user", content=request.message),
            Message(role="assistant", content=response["response"])
        ]

        return ChatResponse(
            response=response["response"],
            chat_history=chat_history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{session_id}/", response_model=ChatResponse)
async def chat_with_session(session_id: str, request: ChatRequest):
    """Handle chat requests with session management."""
    try:
        # Create new session if it doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "conversation": get_chat_chain(
                    model=request.model,
                    memory_length=request.memory_length,
                    persona=request.persona
                ),
                "history": [],
                "start_time": datetime.now()
            }

        session = chat_sessions[session_id]
        conversation = session["conversation"]

        # Get response from conversation chain
        response = conversation(request.message)

        # Update chat history
        new_messages = [
            Message(role="user", content=request.message),
            Message(role="assistant", content=response["response"])
        ]
        session["history"].extend(new_messages)

        return ChatResponse(
            response=response["response"],
            chat_history=session["history"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}/")
async def clear_chat_session(session_id: str):
    """Clear a chat session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": "Chat session cleared successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/chat/{session_id}/history/", response_model=ChatHistory)
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    if session_id in chat_sessions:
        return ChatHistory(messages=chat_sessions[session_id]["history"])
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)