from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

def load_nli_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

def load_nli_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,  
        trust_remote_code=True
    )
    return model

def get_nli_score(model, tokenizer, context, response):
    inputs = tokenizer(context, response, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=1)
    return probs 
