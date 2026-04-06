import torch
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Kingsoft-LLM/QZhou-Embedding", trust_remote_code=True)
texts = ["Hello world"]
embs = model.encode(texts)
print(embs.shape)
