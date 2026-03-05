import logging
import math

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
        """accepts batched token ids and returns token embeddings"""
        batch_size, seq_len = x.shape
        assert seq_len <= self.block_size, (
            f"seq_len {seq_len} exceeds block_size {self.block_size}"
        )

        positions = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        x = self.dropout(token_embeddings + position_embeddings)

        return x


class Transformer(torch.nn.Module):
    def __init__(self, d_model=embed_dim, n_heads=12, dropout=0.1, n_layers=12):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.n_layers = n_layers

        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, dropout, block_size)
        self.blocks = torch.nn.ModuleList()

        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, n_heads, dropout))

        self.final_layer_norm = LayerNorm(d_model, True)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.embedding_layer.token_embedding.weight = self.lm_head.weight  # token embedding weights are tied to the final linear layer weights (reduces parameters significantly)

        self._init()

    def _init(self):
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("Wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        self.to(device)


    def _init_weights(self, module):
        """initialise all weights with Normal(0, 0.02), zeros for biases"""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.embedding_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive token generation.
        idx: LongTensor of shape (B, T) — starting context token indices
        Returns LongTensor of shape (B, T + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_beam(self, idx, max_new_tokens, beam_width=3):
        """
        Beam search generation. idx: (1, T). Returns (1, T + max_new_tokens).
        """
        self.eval()
        beams = [(idx[0].clone(), 0.0)]  # (seq, log_prob)
        for _ in range(max_new_tokens):
            candidates = []
            for seq, score in beams:
                ctx = seq.unsqueeze(0)
                ctx = ctx if ctx.size(1) <= block_size else ctx[:, -block_size:]
                logits = self(ctx)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                top_logprob, top_idx = log_probs[0].topk(beam_width)
                for i in range(beam_width):
                    new_seq = torch.cat([seq, top_idx[i : i + 1]])
                    candidates.append((new_seq, score + top_logprob[i].item()))
            candidates.sort(key=lambda x: -x[1])
            beams = candidates[:beam_width]
        return beams[0][0].unsqueeze(0)

    # def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.95)):
    #     # 2D params (weight matrices, embeddings) get weight decay
    #     # 1D params (biases, LayerNorm scale/shift) do not
    #     decay_params = [
    #         p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2
    #     ]
    #     nodecay_params = [
    #         p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2
    #     ]
    #     optim_groups = [
    #         {"params": decay_params, "weight_decay": weight_decay},
    #         {"params": nodecay_params, "weight_decay": 0.0},
    #     ]
    #     # use fused AdamW on CUDA if available (faster kernel)
    #     use_fused = (
    #         "fused" in inspect.signature(torch.optim.AdamW).parameters
    #         and device.type == "cuda"
    #     )
    #     optimizer = torch.optim.AdamW(
    #         optim_groups,
    #         lr=learning_rate,
    #         betas=betas,
    #         **(dict(fused=True) if use_fused else {}),
    #     )
    #     return optimizer


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
