import time
import pickle
import numpy as np
import torch
from model import Config, GPT

# --- Settings ---
batch_size = 16
max_iters = 3000
eval_interval = 250
eval_iters = 50
learning_rate = 1e-3

torch.manual_seed(42)

meta = pickle.load(open("meta.pkl", "rb"))
itos = meta["itos"]
cfg = Config(meta["vocab_size"])

train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - cfg.block_size - 1, batch_size)
    x = torch.stack([torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+cfg.block_size].astype(np.int64)) for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = [model(*get_batch(split))[1].item() for _ in range(eval_iters)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

def sample(n=200):
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, n)[0].tolist()
    model.train()
    return "".join(itos[i] for i in out)

model = GPT(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
best_val = float("inf")
t0 = time.time()

for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"\nstep {it} | train {losses['train']:.3f} | val {losses['val']:.3f}")
        print("-" * 40)
        print(sample())
        print("-" * 40)
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "vocab_size": cfg.vocab_size}, "ckpt.pt")
            print("saved checkpoint")

    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 10 == 0:
        dt = (time.time() - t0) / (it + 1)
        print(f"iter {it} | loss {loss.item():.3f} | {dt:.2f}s/iter", end="\r")