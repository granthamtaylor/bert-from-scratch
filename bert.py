import math

import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

# the shape of a batch / input
EmbeddingsTensor = TensorType['batch', 'd_model', 'seq_len']
AttentionTensor = TensorType['batch', 'n_heads', 'seq_len', 'seq_len']
MultiHeadTensor = TensorType['batch', 'n_heads', 'seq_len', '3x_d_head']
AttendedValueTensor = TensorType['batch', 'n_heads', 'seq_len', 'd_head']
HeadTensor = TensorType['batch', 'n_heads', 'seq_len', 'd_head']

class Encoder:

    value = 'hello!'

class Embedding:

    def __init__(self):

        pass

class WordEmbedder:

    def __init__(self):

        pass

class EncoderBlockGroup(nn.Module):

    def __init__(
        self,
        n_blocks: int=12,
        d_model: int=512,
        n_heads: int=8,
        d_hidden: int=1024,
        dropout: float=0.0
    ):

        super().__init__()

        self.n_blocks = n_blocks

        layers = []
        for _ in range(n_blocks):
            block = EncoderBlock(d_model, n_heads, d_hidden, dropout)
            layers.append(block)

        self.layers = nn.ModuleList(layers)

    @typechecked
    def forward(self, X: EmbeddingsTensor) -> EmbeddingsTensor:

        for layer in self.layers:
            X = layer(X)

        return X



class EncoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        dropout: float,
    ):

        super().__init__()

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_hidden: int = d_hidden

        self.attn = MultiHeadAttention(d_model, n_heads)

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_model),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @typechecked
    def forward(self, X: EmbeddingsTensor) -> EmbeddingsTensor:

        # residual layer after attention
        X += self.dropout( self.attn(X) )

        # residual layer after MLP
        X += self.dropout( self.mlp(X) )
        X = self.norm(X)

        return X


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
    ):

        super().__init__()

        assert d_model % n_heads == 0 ,\
            'model dimension is not compatible with number of heads'

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.scalar: float = ( 1 / math.sqrt(d_model) )
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)
        self.x3_d_heads = 3 * ( d_model // n_heads )

    @typechecked
    def forward(self, X: EmbeddingsTensor) -> EmbeddingsTensor:

        d_batch, d_seq, d_embed = X.size()

        assert d_embed == self.d_model ,\
            'model dimension not compatible with tensor embedding dimension'

        context: MultiHeadTensor = (
            self.qkv(X)
            .reshape(d_batch, d_seq, self.n_heads, self.x3_d_heads)
            .permute(0, 2, 1, 3)
        )

        output: EmbeddingsTensor = self.fc(
            self.attend(context)
            .permute(0, 2, 1, 3)
            .reshape(d_batch, d_seq, d_embed)
        )
        
        return output

    @typechecked
    def attend(self, context: MultiHeadTensor) -> AttendedValueTensor:

        query: HeadTensor
        key: HeadTensor
        value: HeadTensor

        (query, key, value) = context.chunk(3, dim=-1)
        attention: AttentionTensor = query.matmul(key.transpose(-2, -1)).mul(self.scalar)
        output: AttendedValueTensor = torch.softmax(attention, dim=-1).matmul(value)

        return output

if __name__ == '__main__':

    bert = EncoderBlockGroup()
    x: EmbeddingsTensor = torch.rand((32, 300, 512))
    print(bert)
    y = bert(x)