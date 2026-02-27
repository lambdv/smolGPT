import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import nn

prints = True
v = nn.Vectorizer()
t = nn.TransformerBlock()


@pytest.mark.parametrize(
    "text", ["hi how are you", "hello how are you", "good morning how are you"]
)
def test_masked_attention(text):
    v.to_vec(text)
    x = v.to_vec(text)
    assert x is not None
    y = t._self_attention(x)
    assert y is not None


def test_transformer_block_forward():
    text = "hi how are you"
    v.to_vec(text)
    x = v.to_vec(text)
    assert x is not None
    y = t.forward(x)
    print(y)
    assert y is not None
    assert y.shape == x.shape
