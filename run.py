import pickle
import torch

meta = pickle.load(open("meta.pkl", "rb"))
stoi, itos = meta["stoi"], meta["itos"]

cfg = Config(meta["vocab_size"])
model = GPT(cfg)
ckpt = torch.load("ckpt.pt")
model.load_state_dict(ckpt["model"])
model.eval()

def generate(prompt, n=300, temperature=0.8):
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        ids = [0]
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, n, temperature)[0].tolist()
    return "".join(itos[i] for i in out)

print(generate("The "))