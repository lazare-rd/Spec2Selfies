from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

cache_dir = "./saved_models"

tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large", cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained("zjunlp/MolGen-large", cache_dir=cache_dir)

