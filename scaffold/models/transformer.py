"""Tiny causal transformer for game playing.
Obs -> d_model projection, causal self-attention over trajectory, policy head.

Config is minimal: n_layer=2-4, d_model=64-128, n_head=2-4.
Sequence = trajectory of T steps; each step is (obs_emb + pos). No tokenizer.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, obs_dim, n_actions, d_model=128, n_layer=3, n_head=4, max_len=64, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.d_model = d_model
        self.max_len = max_len

        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_actions)

        self.register_buffer("causal_mask", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())

    def forward(self, obs_seq, mask=None):
        """
        obs_seq: (B, T, obs_dim)  float
        mask:    (B, T)  0/1  (1=valid)  optional
        Returns logits: (B, T, n_actions)
        """
        B, T, _ = obs_seq.shape
        x = self.obs_proj(obs_seq)  # (B,T,d)
        pos = torch.arange(T, device=obs_seq.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)

        causal = self.causal_mask[:T, :T]  # (T,T) True=masked
        # padding: key_padding_mask True means ignore
        kpm = None if mask is None else (mask == 0)
        x = self.encoder(x, mask=causal, is_causal=False, src_key_padding_mask=kpm)
        x = self.ln(x)
        return self.head(x)

    def act(self, obs_seq):
        """Single-trajectory inference: obs_seq (T, obs_dim) -> action int."""
        self.eval()
        with torch.no_grad():
            obs = torch.from_numpy(obs_seq).float().unsqueeze(0) if hasattr(obs_seq, 'shape') and obs_seq.ndim==2 else obs_seq.unsqueeze(0)
            logits = self(obs)  # (1,T,A)
            return int(logits[0, -1].argmax().item())

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
