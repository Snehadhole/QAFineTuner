from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def get_model(model_name = "google/flan-t5-base"):
  original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
  return original_model

def get_tokenizer(model_name = "google/flan-t5-base"):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  return  tokenizer
