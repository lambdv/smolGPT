import pytest
import torch

import model

prints = True
# v = nn.Vectorizer()


def test_transformer_init():
    m = model.Transformer()
    assert m is not None


def test_transformer_forward_loss():
    m = model.Transformer()
    idx = torch.randint(0, 100, (2, 16), device=model.device)
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, 16, model.vocab_size)
    assert loss.dim() == 0 and loss.item() >= 0


def test_transformer_generate():
    m = model.Transformer()
    idx = torch.randint(0, 100, (1, 4), device=model.device)
    out = m.generate(idx, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 4 + 5)
    assert out[:, :4].equal(idx)


def test_transformer_generate_beam():
    m = model.Transformer()
    idx = torch.randint(0, 100, (1, 4), device=model.device)
    out = m.generate_beam(idx, max_new_tokens=5, beam_width=3)
    assert out.shape == (1, 4 + 5)
    assert out[:, :4].equal(idx)
