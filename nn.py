import logging
import math

# v = Vectorizer()
# x = v.to_vec("hi")
# # print(x.shape)
# t = TransformerBlock()
# t._self_attention(x)
import torch
import torch.nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel

logging.getLogger("transformers").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

vocab_size = 50304  # number of unique tokens in the vocabulary
embed_dim = 768  # number of dimensions in the embedding space
seq_len = 3  # sequence length
block_size = 1024  # maximum sequence length of the model


class EmbeddingLayer(torch.nn.Module):
    def __init__(
        self, vocab_size=vocab_size, d_model=embed_dim, dropout=0.1, block_size=1024
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        self.token_embedding = torch.nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(self.block_size, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.shape
        assert seq_len <= self.block_size, f"seq_len {seq_len} exceeds block_size {self.block_size}"

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        
        x = self.dropout(token_embeddings + position_embeddings)
        
        return x


class Transformer(torch.nn.Module):
    def __init__(self, d_model=embed_dim, n_heads=12, dropout=0.1, n_layers=12):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, dropout, block_size)

        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_layer_norm = LayerNorm(d_model, True)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.embedding_layer.token_embedding.weight = self.lm_head.weight # token embedding weights are tied to the final linear layer weights (reduces parameters significantly)
        self.to(device)

    def forward(self, x):
        x = self.embedding_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def predict(self, text):
        logits = self.forward(text)
        return logits.argmax(dim=-1)


class Vectorizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to(device)

    def to_vec(self, text):
        inputs = {
            "input_ids": self.tokenizer.encode(text, return_tensors="pt").to(device)
        }
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs[0]


def attention(K: torch.Tensor, Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return F.softmax(Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5), dim=-1) @ V
    # attention is quadraitc (polynomial variable form) where vector^t * matrix * vector is quadratic aka T * Q * V


def linear_projection(token_embedding, Wq, Wk, Wv):
    """project token embedding as as its learnt kqv for this layer"""
    K = Wk(token_embedding)
    Q = Wq(token_embedding)
    V = Wv(token_embedding)
    return K, Q, V


def mased_self_attention(x, Wq, Wk, Wv, Wo, n_heads, d_head):
    batchs, token_sequences, dimension = x.shape
    K, Q, V = linear_projection(x, Wq, Wk, Wv)

    Q = Q.view(batchs, token_sequences, n_heads, d_head).transpose(1, 2)
    K = K.view(batchs, token_sequences, n_heads, d_head).transpose(1, 2)
    V = V.view(batchs, token_sequences, n_heads, d_head).transpose(1, 2)

    attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)

    mask = torch.tril(
        torch.ones(token_sequences, token_sequences, device=x.device)
    ).view(1, 1, token_sequences, token_sequences)  # triangle shaped matrix
    attention_scores = attention_scores.masked_fill(mask == 0, -1e9)  # apply mask

    weights = torch.softmax(attention_scores, dim=-1)
    weighted_sums = weights @ V

    out = (
        weighted_sums.transpose(1, 2)
        .contiguous()
        .view(batchs, token_sequences, dimension)
    )
    return Wo(out)


class LayerNorm(torch.nn.Module):
    def __init__(self, n_dim, bias=True, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n_dim))
        self.bias = torch.nn.Parameter(torch.zeros(n_dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


#
def MLP(d_model=embed_dim, dropout=0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(d_model, d_model * 4, bias=False),  # eg: 768 -> 3072
        torch.nn.GELU(),  # non-linear activation function
        torch.nn.Linear(d_model * 4, d_model, bias=False),  # eg: 3072 -> 768
        torch.nn.Dropout(
            dropout
        ),  # randomly drop out some tokens to prevent overfitting
    )


class TransformerBlock(torch.nn.Module):
    # transformer has 2 subsections: attention and MLP, each section has a learnt normal layer and residual connections
    def __init__(self, d_model=embed_dim, n_heads=12, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout

        # self.norm_1 = torch.nn.Parameter(torch.ones(embed_dim))
        # self.norm_2 = torch.nn.Parameter(torch.ones(embed_dim))
        self.norm_1 = LayerNorm(embed_dim, True)
        self.norm_2 = LayerNorm(embed_dim, True)

        # residule layers are apart of archecture but not learnt: runtime graph operations apart of each sub section

        # linear NN of input and output size embed_dim:
        # learns a matrix you multiply by token embedding to get KQV
        self.Wq, self.Wk, self.Wv = (
            torch.nn.Linear(embed_dim, embed_dim, bias=False),
            torch.nn.Linear(embed_dim, embed_dim, bias=False),
            torch.nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False)

        self.MLP = MLP(d_model, dropout)

        self.to(device)

    def forward(self, x):
        x = self._residual(x, self.norm_1, self._self_attention)
        x = self._residual(x, self.norm_2, self.MLP)
        return x

    def _residual(self, x, normal, sublayer):
        return x + sublayer(normal(x))

    def _self_attention(self, x):
        d_head = self.d_model // self.n_heads
        return mased_self_attention(
            x, self.Wq, self.Wk, self.Wv, self.Wo, self.n_heads, d_head
        )
