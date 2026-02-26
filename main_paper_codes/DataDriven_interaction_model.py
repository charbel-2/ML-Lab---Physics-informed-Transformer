import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class MLP(nn.Module):

    def __init__(self, d_model, dropout=0.15, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        
    
class SelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.mha(x, x, x, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(x, x, x, is_causal=False)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y
    
class CrossAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, mem, x):
        x = self.mha(x, mem, mem, is_causal=self.causal)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout =0.0, bias = False):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout= dropout,causal= False, bias= bias)
        self.norm1 = LayerNorm(embed_dim, bias=bias)
        self.norm2 = LayerNorm(embed_dim, bias= bias)
        self.mlp = MLP(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )


    def forward(self, x):
        norm1 = self.norm1(x)
        attention = self.attention(norm1)
        x = attention+x
        norm2 = self.norm2(x)
        mlp1 = self.mlp(norm2)
        x = x + mlp1

        return x

# Transformer Decoder Layer with Cross-Attention
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout = 0.0, bias = False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads, dropout= dropout, causal = True, bias = bias)  # Self-attention in decoder
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout= dropout, causal = False, bias =bias)  # Cross-attention with encoder output
        self.norm1 = LayerNorm(embed_dim, bias=bias)
        self.norm2 = LayerNorm(embed_dim, bias=bias)
        self.norm3 = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

    def forward(self, x, decoder_input): 
        # Self-attention within the decoder layer
        norm1= self.norm1(decoder_input)  # Apply normalization before self-attention
        self_attention = self.self_attention(norm1)
        decoder_input= self_attention + decoder_input
        norm2 = self.norm2(decoder_input)
        cross_attention = self.cross_attention(x, norm2)  
        decoder_output = cross_attention  + decoder_input
        norm3 = self.norm3(decoder_output)  # Apply normalization before MLP
        mlp1 = self.mlp(norm3)
        decoder_output = mlp1 + decoder_output

        return decoder_output


class DataAwareLearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_length, mean, std, max_len=352, init_std=1e-6):
        super(DataAwareLearnablePositionalEncoding, self).__init__()
        
        # Ensure mean and std are tensors and match embed_dim
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32).to(device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32).to(device)
        
        # Expand mean and std to match embed_dim
        mean = mean.mean().expand(embed_dim)
        std = std.mean().expand(embed_dim)
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.mean = mean
        self.std = std

        # Learnable positional encodings
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_len, embed_dim).to(device)
        )

        # Data-aware initialization
        nn.init.normal_(self.positional_encoding, mean=mean.mean().item(), std=init_std)

    def forward(self, x):
        # Slice positional encodings to match input sequence length
        pos_enc = self.positional_encoding[:, :x.size(1)]
        
        # Normalize positional encodings to match training data scale
        pos_enc = (pos_enc - self.mean) / self.std
        
        return x + pos_enc


# Learnable Positional Encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=352):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_embeddings[:, :seq_len]

class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model, mean, std,max_len=352):
        super(HybridPositionalEncoding, self).__init__()
        self.learnable_positional_encoding = LearnablePositionalEncoding(d_model, max_len)
        self.Dataawar_encoding = DataAwareLearnablePositionalEncoding(d_model, seq_length= max_len, mean= mean, std= std, max_len= max_len)
        self.d_model = d_model

    def forward(self, x):
        # Standard positional encoding
        seq_len = x.size(1)
        learnable_encoding = self.learnable_positional_encoding(x)  ## added only for performance point of view
        data_encoding = self.Dataawar_encoding(x)
        
        return learnable_encoding  + data_encoding 
    

# Updated EnhancedTransformer class
class EnhancedTransformerData(nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, n_embd, forward_expansion,seq_len, mean, std,dropout = 0.0, bias = False):
        super(EnhancedTransformerData, self).__init__()
        
        self.encoder_wte = nn.Linear(input_dim, n_embd).to(device)
        self.encoder_wpe = nn.Embedding(seq_len, n_embd).to(device)
        
        self.decoder_wte = nn.Linear(input_dim -9, n_embd).to(device)
        self.decoder_wpe = nn.Embedding(seq_len, n_embd).to(device)   ## From the physics informed ones and maybe useless, but comes from the physics informed where it has been added energy and so on.
        
        self.positional_encoding = HybridPositionalEncoding(n_embd, mean, std,).to(device)
        
        self.norm1 = LayerNorm(n_embd, bias=bias).to(device)  ## wrt to MARCO, this can be diffferent from encoder and decoder, here is the same.
        
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)
        
        self.decoder_output = nn.Linear(n_embd, 3, bias= True).to(device)  # Output layer for torque prediction
        
    def DecoderEmbedding(self, decoder_input):
        
        device = decoder_input.device
        seq_len_decoder = decoder_input.size(1)
        
        # Positional embedding
        pos_decoder = torch.arange(0, seq_len_decoder, dtype=torch.long, device=device).unsqueeze(0)
        
        pos_emb_decoder = self.decoder_wpe(pos_decoder)
        tok_emb_decoder = self.decoder_wte(decoder_input)

        # Combine physics-aware embedding and positional embedding
        return   tok_emb_decoder + pos_emb_decoder

    def EncoderEmbeding(self, x):
        
        device = x.device
        seq_len_encoder = x.size(1)
        
        # Positional embedding
        pos_encoder= torch.arange(0, seq_len_encoder, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb_encoder = self.encoder_wpe(pos_encoder)
        tok_emb_encoder = self.encoder_wte(x)

        # Combine physics-aware embedding and positional embedding
        return  tok_emb_encoder + pos_emb_encoder

    def forward(self, x, decoder_input):
        
        x = self.EncoderEmbeding(x)
        decoder_input = self.DecoderEmbedding(decoder_input)
        
        x = self.positional_encoding(x)
        decoder_input = self.positional_encoding(decoder_input)
        
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm1(x)
        
        for layer in self.decoder_layers:
            decoder_output = layer(x, decoder_input)
        decoder_output = self.norm1(decoder_output)
        
        return self.decoder_output(decoder_output)  # Predict based on the last time step
