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

vocab_size = 10
embed_dim = 768  # d_model
seq_len = 3


class Transformer:
    ""

    pass


class Vectorizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.cuda()  # type: ignore

    def to_vec(self, token):
        inputs = self.tokenizer(token, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


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

    mask = torch.tril(torch.ones(seq_len, seq_len))  # triangle shaped matrix
    attention_scores = attention_scores.masked_fill(
        mask == 0, -float("inf")
    )  # apply mask

    weights = torch.softmax(attention_scores, dim=-1)
    weighted_sums = weights @ V

    out = (
        weighted_sums.transpose(1, 2)
        .contiguous()
        .view(batchs, token_sequences, dimension)
    )
    return Wo(out)


class LayerNorm(torch.nn.Module):
    def __init__(self, n_dim, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n_dim))
        self.bias = torch.nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TransformerBlock(torch.nn.Module):
    # transformer has 2 subsections: attention and MLP, each section has a learnt normal layer and residual connections
    def __init__(self, d_model=embed_dim, n_heads=12, dropout=0.1):
        super().__init__()
        # self.norm_1 = torch.nn.Parameter(torch.ones(embed_dim))
        # self.norm_2 = torch.nn.Parameter(torch.ones(embed_dim))
        self.norm_1 = LayerNorm(embed_dim, True)
        self.norm_2 = LayerNorm(embed_dim, True)

        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout

        # residule layers are apart of archecture but not learnt: runtime graph operations apart of each sub section

        # linear NN of input and output size embed_dim:
        # learns a matrix you multiply by token embedding to get KQV
        self.Wq, self.Wk, self.Wv = (
            torch.nn.Linear(embed_dim, embed_dim, bias=False),
            torch.nn.Linear(embed_dim, embed_dim, bias=False),
            torch.nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False)

        self.MLP = ""

    def _self_attention(self, x):
        return mased_self_attention(
            x, self.Wq, self.Wk, self.Wv, self.Wo, self.n_heads, self.d_model
        )
