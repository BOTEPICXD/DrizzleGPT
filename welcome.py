# Project: Chatbot Simulator Repo - Fully Auto Setup Script
# This update fills welcome.py with the actual working code from all files.

# ======= welcome.py =======
import os
import pathlib

# Project structure and content
files = {
    'main.py': '''from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any
import os
from rag_setup import Retriever
from simulation import run_simulation

GEN_BACKEND = os.getenv('GENERATOR','openai')
app = FastAPI(title='Chatbot Simulator')
retriever = Retriever(index_path='faiss_index.pkl', embedding_model='all-MiniLM-L6-v2')

class ChatRequest(BaseModel):
    user_id: Optional[str]
    message: str
    run_sim: Optional[bool] = False
    sim_params: Optional[dict] = None

class ChatResponse(BaseModel):
    reply: str
    sim_output: Optional[Any] = None

@app.post('/chat', response_model=ChatResponse)
async def chat(req: ChatRequest):
    sim_result = None
    if req.run_sim:
        sim_result = run_simulation(req.sim_params or {})

    docs = retriever.get_relevant(req.message, k=4)
    context_text = "\n\n---\n\n".join(docs)

    system = "You are a helpful assistant. Use provided context and simulate reasoning when needed."
    prompt = f"{system}\n\nContext:\n{context_text}\n\nUser: {req.message}\nAssistant:"

    if sim_result:
        prompt += "\n\nSimulation results (structured):\n" + str(sim_result['summary'])

    if GEN_BACKEND == 'openai':
        from generators.openai_gen import generate
    else:
        from generators.hf_gen import generate

    reply = generate(prompt)
    return ChatResponse(reply=reply, sim_output=sim_result)
''',

    'rag_setup.py': '''import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class Retriever:
    def __init__(self, index_path='faiss_index.pkl', embedding_model='all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.model = SentenceTransformer(embedding_model)
        if os.path.exists(index_path):
            with open(index_path,'rb') as f:
                saved = pickle.load(f)
            self.index = saved['index']
            self.docs = saved['docs']
        else:
            self.index = None
            self.docs = []

    def add_documents(self, texts: list):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)
            self.docs = list(texts)
        else:
            self.index.add(embeddings)
            self.docs.extend(texts)
        with open(self.index_path,'wb') as f:
            pickle.dump({'index': self.index, 'docs': self.docs}, f)

    def get_relevant(self, query: str, k: int = 4):
        if self.index is None or len(self.docs) == 0:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        _, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results
''',

    'simulation.py': '''import random
import statistics
from typing import Dict, Any

def run_simulation(params: Dict = None) -> Dict[str, Any]:
    params = params or {}
    steps = int(params.get('steps', 100))
    trials = int(params.get('trials', 1000))
    init = float(params.get('init', 100.0))
    growth = float(params.get('growth', 0.01))
    vol = float(params.get('vol', 0.05))

    final_values = []
    for t in range(trials):
        v = init
        for s in range(steps):
            shock = random.gauss(growth, vol)
            v = max(0.0, v * (1 + shock))
        final_values.append(v)

    mean = statistics.mean(final_values)
    median = statistics.median(final_values)
    stdev = statistics.stdev(final_values)

    summary = (
        f"Ran {trials} trials of a {steps}-step stochastic growth simulation. "
        f"Initial value {init}, mean final value {mean:.2f}, median {median:.2f}, stdev {stdev:.2f}."
    )

    return {
        'params': params,
        'mean': mean,
        'median': median,
        'stdev': stdev,
        'summary': summary,
        'samples': final_values[:20]
    }
''',

    'requirements.txt': '''fastapi
uvicorn[standard]
langchain
transformers
accelerate
datasets
sentence_transformers
faiss-cpu
pydantic
python-multipart
aiofiles
openai
tiktoken
requests''',

    'README.md': '# Chatbot Simulator\nRun the Chatbot Simulator with FastAPI + RAG + Simulation Engine.',

    'example_usage.md': '# Example Usage\nSee README for instructions.'
}

# Generators folder content
generators_files = {
    'generators/openai_gen.py': '''import os
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate(prompt: str) -> str:
    resp = openai.ChatCompletion.create(
        model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),
        messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':prompt}],
        max_tokens=600,
        temperature=0.2,
    )
    return resp['choices'][0]['message']['content'].strip()
''',

    'generators/hf_gen.py': '''from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

_model = None
_tokenizer = None

def _init():
    global _model, _tokenizer
    if _model is None:
        model_name = os.getenv('HF_MODEL','facebook/llama-2-7b-chat')
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)

def generate(prompt: str) -> str:
    _init()
    gen = pipeline('text-generation', model=_model, tokenizer=_tokenizer)
    out = gen(prompt, max_new_tokens=512, do_sample=False)
    return out[0]['generated_text']
'''
}

# Create folders
pathlib.Path('generators').mkdir(exist_ok=True)

# Write files
for fname, content in files.items():
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)

for fname, content in generators_files.items():
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)

print("All files created! Run 'pip install -r requirements.txt' to install dependencies.")
print("Then set environment variables and run 'uvicorn main:app --reload --port 8000'")
