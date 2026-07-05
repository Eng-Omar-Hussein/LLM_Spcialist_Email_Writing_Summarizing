import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Global variable to hold the model instance
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    try:
        print("Downloading Qwen 2.5 0.5B GGUF directly from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )
        print("Loading model into application memory...")
        # n_ctx=2048 limits the maximum token memory structure to save RAM
        llm = Llama(
            model_path=model_path, 
            n_ctx=2048, 
            n_threads=4,
            # Explicitly declaring chat format can help ensure correct template mapping
            chat_format="chatml" 
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Critical error loading model: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# --- Direct Output System Prompts ---
SYSTEM_PROMPT_WRITE = (
    "You are an expert email writing engine. Draft an email based strictly on the user's request. "
    "CRITICAL INSTRUCTION: Output ONLY the raw email content. Do not include any conversational filler, greetings to the user, explanations, or acknowledgments."
)

SYSTEM_PROMPT_SUMMARIZE = (
    "You are an expert email summarization engine. Summarize the provided text. "
    "CRITICAL INSTRUCTION: Output ONLY the summary. Do not include any conversational filler, introductions like 'Here is the summary:', explanations, or acknowledgments."
)

class SummarizeRequest(BaseModel):
    email_text: str
    custom_instruction: str = "Summarize clearly with main points."

class WriteRequest(BaseModel):
    user_prompt: str

# --- Endpoints ---
@app.post("/summarize")
async def summarize_email(request: SummarizeRequest):
    # Constructing the chat history payload
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
        {"role": "user", "content": request.email_text}
    ]
    return {"result": execute_inference(messages)}

@app.post("/write")
async def write_email(request: WriteRequest):
    # Constructing the chat history payload
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_WRITE},
        {"role": "user", "content": request.user_prompt}
    ]
    return {"result": execute_inference(messages)}

def execute_inference(messages: list) -> str:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is initializing.")
    
    # Use create_chat_completion to automatically apply the correct chat template (ChatML)
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.1
    )
    
    # Extract the response from the standardized chat completion dictionary structure
    raw_text = output["choices"][0]["message"]["content"]
    return raw_text.strip()
