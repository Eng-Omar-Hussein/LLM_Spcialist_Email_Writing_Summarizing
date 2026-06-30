import httpx
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = "qwen2.5:1.5b" 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically pulls the model into Ollama on startup.
    # Timeout set to 600s to allow time for the ~1GB download over the network.
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            print(f"Checking and pulling model {MODEL_NAME} if not present...")
            await client.post(f"{OLLAMA_URL}/api/pull", json={"name": MODEL_NAME})
            print("Model is ready.")
        except Exception as e:
            print(f"Warning: Could not pull model from Ollama: {e}")
    yield

app = FastAPI(lifespan=lifespan)

class SummarizeRequest(BaseModel):
    email_text: str
    system_prompt: str = "Summarize the following email clearly. Extract main points and action items."

class WriteRequest(BaseModel):
    user_prompt: str

@app.post("/summarize")
async def summarize_email(request: SummarizeRequest):
    prompt = f"System Instruction: {request.system_prompt}\n\nEmail Content:\n{request.email_text}"
    return await query_model(prompt)

@app.post("/write")
async def write_email(request: WriteRequest):
    prompt = f"System Instruction: Write a professional email satisfying the following request.\n\nUser Request:\n{request.user_prompt}"
    return await query_model(prompt)

async def query_model(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False 
    }
    
    # Timeout set to 300s to allow CPU inference time.
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return {"result": data.get("response", "").strip()}
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"LLM backend error: {str(e)}")
