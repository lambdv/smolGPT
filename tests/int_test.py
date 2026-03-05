import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import model

prints = True
# v = nn.Vectorizer()

# def test_untrained_pass(text):
#     model = nn.Transformer()
#     # Tokenize text to token IDs
#     token_ids = model.tokenizer.encode(text, return_tensors="pt").to(nn.device)
#     # Pass token IDs to model forward method
#     logits, loss = model(token_ids)
