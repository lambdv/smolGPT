# smolgpt
a small generative pre-trained transformer model implemented in pytorch.
![smolgpt](image.png)
```python
class Transformer(torch.nn.Module):
    def __init__(self, d_model=embed_dim, n_heads=12, dropout=0.1, n_layers=12):
        super().__init__()
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.n_layers = n_layers
    
        # embedding layer
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, dropout, block_size)
        # transformer blocks
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, n_heads, dropout))
        # final layer norm
        self.final_layer_norm = LayerNorm(d_model, True)
        # linear projection to vocab
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

        self.embedding_layer.token_embedding.weight = self.lm_head.weight 
        
        _init(self)

        self.to(device)
```

## getting started
1. install dependencies
```bash
python -m venv .venv
pip install -r requirements.txt
./.venv/Scripts/activate
```

2. train
```bash
python prepare.py shakespeare_char # prepare data
python train.py # train model weights
```

3. generate text
```bash
python use.py "Once upon a time"
```

