import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Any

# ---------------- App ----------------
app = FastAPI(title="DrizzleGPT")

# ---------------- Globals ----------------
retriever = None
GEN_BACKEND = os.getenv("GENERATOR", "openai")

# ---------------- Models ----------------
class ChatRequest(BaseModel):
    user_id: Optional[str]
    message: str
    run_sim: Optional[bool] = False
    sim_params: Optional[dict] = None

class ChatResponse(BaseModel):
    reply: str
    sim_output: Optional[Any] = None

# ---------------- Async OpenAI wrapper ----------------
async def async_generate(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def sync_call():
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are Drizzle, a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error generating response: {str(e)}]"

    return await asyncio.to_thread(sync_call)

# ---------------- POST /chat ----------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global retriever
    try:
        # Lazy-load retriever
        if retriever is None:
            from rag_setup import Retriever
            retriever = Retriever(
                index_path="faiss_index.pkl",
                embedding_model="all-MiniLM-L6-v2"
            )

        # Optional simulation
        sim_result = None
        if req.run_sim:
            try:
                from simulation import run_simulation
                sim_result = run_simulation(req.sim_params or {})
            except Exception as e:
                sim_result = {"error": str(e)}

        # Get relevant docs
        docs = retriever.get_relevant(req.message, k=4)
        context_text = "\n\n---\n\n".join(docs)

        # Compose prompt
        prompt = f"You are Drizzle, a helpful assistant.\n\nContext:\n{context_text}\n\nDrizzle: {req.message}"
        if sim_result:
            prompt += "\n\nSimulation results:\n" + str(sim_result.get("summary",""))

        # Generate reply
        reply = await async_generate(prompt)
        return ChatResponse(reply=reply, sim_output=sim_result)

    except Exception as e:
        # Always return valid JSON
        return ChatResponse(reply=f"[Error generating response: {str(e)}]", sim_output=None)

# ---------------- Health check ----------------
@app.get("/kaithheathcheck")
async def health_check():
    return {"status": "ok", "message": "DrizzleGPT is healthy!"}

# ---------------- HTML frontend ----------------
@app.get("/", response_class=HTMLResponse)
async def chat_page():
    return """<!DOCTYPE html>
<html>
<head>
<title>DrizzleGPT</title>
<style>
body { font-family: 'Segoe UI', sans-serif; margin:0; padding:0; background:#121212; color:#e0e0e0; }
.container { max-width:800px; margin:0 auto; padding:20px; }
h1 { text-align:center; color:#1de9b6; }
#chat { border:1px solid #333; border-radius:10px; padding:15px; height:500px; overflow-y:auto; background:#1e1e1e; }
.message { padding:10px 15px; margin:8px 0; border-radius:20px; max-width:70%; word-wrap:break-word; opacity:0; transform:translateY(10px); animation:fadeIn 0.3s forwards; }
@keyframes fadeIn { to { opacity:1; transform:translateY(0); } }
.user { background: linear-gradient(135deg,#2979ff,#448aff); color:white; margin-left:auto; text-align:right; }
.bot { background: linear-gradient(135deg,#00c853,#00e676); color:white; margin-right:auto; text-align:left; }
.sim { background: linear-gradient(135deg,#7c4dff,#b388ff); color:white; font-style:italic; text-align:center; margin:8px auto; max-width:90%; }
.timestamp { font-size:0.7em; color:#aaa; margin-top:2px; }
#input-container { display:flex; margin-top:15px; }
#user-input { flex:1; padding:10px; border-radius:20px; border:none; outline:none; background:#1e1e1e; color:#e0e0e0; }
#send { margin-left:10px; padding:10px 20px; border-radius:20px; border:none; background:#1de9b6; color:#121212; cursor:pointer; font-weight:bold; }
#send:hover { background:#00bfa5; }
#quick-replies { margin-top:10px; display:flex; gap:10px; flex-wrap:wrap; }
.quick-btn { background:#333; border:none; padding:5px 10px; border-radius:15px; cursor:pointer; color:#e0e0e0; }
.quick-btn:hover { background:#444; }
#typing-indicator { font-style:italic; color:#aaa; margin-top:5px; }
</style>
</head>
<body>
<div class="container">
<h1>DrizzleGPT</h1>
<div id="chat"></div>
<div id="typing-indicator" style="display:none;">Drizzle is typing...</div>
<div id="quick-replies"></div>
<div id="input-container">
<input type="text" id="user-input" placeholder="Type a message..." />
<button id="send">Send</button>
</div>
</div>
<script>
const chatDiv = document.getElementById('chat');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send');
const typingIndicator = document.getElementById('typing-indicator');
const quickReplies = document.getElementById('quick-replies');

const quickOptions = ["Hello!", "Run simulation", "Help", "Explain RAG"];
quickOptions.forEach(opt => {
    const btn = document.createElement('button');
    btn.className = 'quick-btn';
    btn.textContent = opt;
    btn.onclick = () => { input.value = opt; sendMessage(); };
    quickReplies.appendChild(btn);
});

function appendMessage(sender,text) {
    const div = document.createElement('div');
    div.className = 'message ' + sender;
    let label;
    if(sender==='bot') label='Drizzle';
    else if(sender==='sim') label='Simulation';
    else label='You';
    const time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    div.innerHTML=`<strong>${label}</strong>: ${text}<div class="timestamp">${time}</div>`;
    chatDiv.appendChild(div);
    chatDiv.scrollTop = chatDiv.scrollHeight;
}

async function sendMessage() {
    const msg = input.value;
    if(!msg) return;
    appendMessage('user',msg);
    input.value='';
    typingIndicator.style.display='block';

    try {
        const response = await fetch('/chat',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({user_id:null,message:msg,run_sim:false,sim_params:null})
        });
        const data = await response.json();
        typingIndicator.style.display='none';
        appendMessage('bot',data.reply);
        if(data.sim_output) appendMessage('sim',JSON.stringify(data.sim_output));
    } catch(err) {
        typingIndicator.style.display='none';
        appendMessage('bot',`[Error sending request: ${err}]`);
    }
}

sendBtn.onclick=sendMessage;
input.addEventListener('keypress',e=>{if(e.key==='Enter') sendMessage();});
</script>
</body>
</html>
"""
