import sys

import torch

import model

checkpoint_path = "checkpoints/model.pt"
max_new_tokens = 100
temperature = 0.8


def run(prompt="Hello, world", beam=False):
    m = model.Transformer()
    if __name__ == "__main__" and hasattr(sys, "argv") and len(sys.argv) > 1:
        args = [a for a in sys.argv[1:] if a != "--beam"]
        beam = "--beam" in sys.argv
        prompt = " ".join(args) if args else prompt
    if __name__ == "__main__":
        try:
            m.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
        except FileNotFoundError:
            print(f"No checkpoint at {checkpoint_path}. Run train.py first.")
            return
    idx = torch.tensor(
        [m.tokenizer.encode(prompt)], dtype=torch.long, device=model.device
    )
    out = m.generate_beam(idx, max_new_tokens) if beam else m.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    print(m.tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    run()
