# llm_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

@st.cache_resource(show_spinner="Loading DeepSeek model...")
def load_deepseek_model(model_id="deepseek-ai/deepseek-coder-1.3b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
