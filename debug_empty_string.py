import torch
from sentence_transformers import SentenceTransformer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("Kingsoft-LLM/QZhou-Embedding", trust_remote_code=True)

try:
    print("Encoding normal...")
    emb = model.encode(["Hello world"])
    print("Normal done")
    
    print("Encoding empty...")
    emb2 = model.encode([""])
    print("Empty done")
    
    print("Encoding NaN string...")
    emb3 = model.encode(["nan", "NaN"])
    print("NaN string done")
    
except Exception as e:
    import traceback
    traceback.print_exc()

