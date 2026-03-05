import os

import torch

import model

device = model.device
block_size = model.block_size
batch_size = 8
context_len = 128
max_steps = 5000
lr = 3e-4
checkpoint_path = "checkpoints/model.pt"


def train(m=None):
    m = m or model.Transformer()
    m.train()
    data_path = "data/train.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Put training text in {data_path}")
    text = open(data_path).read()
    tokens = torch.tensor(m.tokenizer.encode(text), dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(max_steps):
        starts = torch.randint(0, len(tokens) - context_len, (batch_size,))
        idx = torch.stack([tokens[s : s + context_len] for s in starts.tolist()]).to(device)
        _, loss = m(idx, targets=idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"step {step + 1} loss {loss.item():.4f}")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(m.state_dict(), checkpoint_path)
    print(f"Saved {checkpoint_path}")


if __name__ == "__main__":
    train()
