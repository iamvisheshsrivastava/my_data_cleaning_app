# llm_utils.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_id = "google/flan-t5-base"  # Lightweight, good instruction follower
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.to("cpu")  # Safe for local CPU usage

    return tokenizer, model
