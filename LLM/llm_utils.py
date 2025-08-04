# llm_utils.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_id = "google/flan-t5-base" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.to("cpu") 

    return tokenizer, model
