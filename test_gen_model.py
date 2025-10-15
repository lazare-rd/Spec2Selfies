from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Inference running on {device}")

tokenizer = AutoTokenizer.from_pretrained("./saved_models/models--zjunlp--MolGen-large/snapshots/3c6e8fb91a783853a3782552a85efc4df6f96d0a")
model = AutoModelForSeq2SeqLM.from_pretrained("./saved_models/models--zjunlp--MolGen-large/snapshots/3c6e8fb91a783853a3782552a85efc4df6f96d0a")
model.to(device)

print (model)
