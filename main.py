import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any

# GENERATOR backend
GEN_BACKEND = os.getenv("GENERATOR", "openai")

app = FastAPI(title="DrizzleGPT")

# Lazy-loaded globals
retriever = None
generate_func = None

# ---- Request/Response Models ----
class ChatRequest(BaseModel):
    user_id: Optional[str]
    message: str
    run_sim: Optional[bool] = False
    sim_params: Optional[dict] = None

class ChatResponse(BaseModel):
    reply: str
    sim_output: Optional[Any] = None

# ---- POST /chat ----
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global retriever, generate_func

    # Lazy-load generator function
    if generate_func is None:
        if GEN_BACKEND == "openai":
            from generators.openai_gen import generate
        else:
            from generators.hf_gen import generate
        generate_func = generate

    # Lazy-load retriever
    if retriever is None:
        from rag_setup import Retriever
        retriever = Retriever(
            index_path="faiss_index.pkl",
            embedding_model="all-MiniLM-L6-v2",
        )

    # Optional simulation
    sim_result = None
    if req.run_sim:
        from simulation import run_simulation
        sim_result = run_simulation(req.sim_params or {})

    # Get relevant docs
    docs = retriever.get_relevant(req.message, k=4)
    context_text = "\n\n---\n\n".join(docs)

    # Compose prompt
    system_prompt = "You are a helpful assistant. Use provided context and simulate reasoning when needed."
    prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nUser: {req.message}\nDrizzle:"

    if sim_result:
        prompt += "\n\nSimulation results (structured):\n" + str(sim_result.get("summary", ""))

    # Generate reply
    reply = generate_func(prompt)

    return ChatResponse(reply=reply, sim_output=sim_result)

# ---- Root endpoint ----
@app.get("/")
async def root():
    return {"message": "DrizzleGPT is running. Use POST /chat to talk to Drizzle!"}
