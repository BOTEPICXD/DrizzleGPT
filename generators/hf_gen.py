from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
