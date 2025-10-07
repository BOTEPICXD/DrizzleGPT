import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/chat"

def chat_with_bot(message, run_sim=False, steps=50, trials=100):
    payload = {
        "message": message,
        "run_sim": run_sim,
        "sim_params": {"steps": steps, "trials": trials}
    }
    response = requests.post(API_URL, json=payload).json()
    reply = response.get("reply", "")
    sim_output = response.get("sim_output", "")
    if sim_output:
        return f"{reply}\n\nSimulation:\n{sim_output['summary']}"
    return reply

iface = gr.Interface(
    fn=chat_with_bot,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your message here..."),
        gr.Checkbox(label="Run Simulation?"),
        gr.Number(label="Simulation Steps", value=50),
        gr.Number(label="Simulation Trials", value=100)
    ],
    outputs="text",
    title="DrizzleGPT"
)

iface.launch()
