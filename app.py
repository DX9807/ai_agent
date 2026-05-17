# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
from agent import GroqAgent, StreamingGroqAgent

app = FastAPI(title="Groq Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = GroqAgent(verbose=False)

class ChatRequest(BaseModel):
    message: str
    clear_history: bool = False

class ChatResponse(BaseModel):
    response: str
    history_length: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for the agent.
     - `message`: The user message to send to the agent.
     - `clear_history`: Whether to clear conversation history before processing the message.
    """
    if request.clear_history:
        agent.reset_memory()
    
    response = agent.chat(request.message)
    history = agent.get_conversation_history()
    
    return ChatResponse(
        response=response,
        history_length=len(history)
    )

@app.get("/history")
async def get_history():
    """
    Get conversation history.
     - Returns the full conversation history as a list of messages.
    """
    return {"history": agent.get_conversation_history()}

@app.post("/clear")
async def clear_history():
    """
    Clear conversation history.
     - Clears the conversation history.
    """
    agent.reset_memory()
    return {"message": "History cleared"}

# Create stream endpoint for real-time responses using StreamingGrokAgent


stream_chat_agent = StreamingGroqAgent()

@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream chat responses in real-time.
     - `message`: The user message to send to the agent.
     - Streams responses as they are generated.
    """
    # data = await request.json()
    message = request.message
    clear_history = request.clear_history

    if clear_history:
        stream_chat_agent.reset_memory()

    async def response_generator():
        for response in stream_chat_agent.stream_chat(message):
            yield f"data: {response}\n\n"
            await asyncio.sleep(0.1)  # Simulate delay for streaming

    return StreamingResponse(response_generator(), media_type="text/event-stream")
