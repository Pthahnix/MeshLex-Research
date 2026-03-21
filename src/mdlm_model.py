"""Full-scale Masked Discrete Language Model for mesh token generation.

Architecture: Bidirectional Transformer Encoder with:
- Token embedding (vocab_size = K + 1 for MASK)
- Positional embedding
- Level embedding (L1/L2/L3 awareness via position mod tokens_per_patch)
- Continuous time embedding (MLP: t -> d_model)
- Output head predicts K real tokens (not MASK)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullMDLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1025,      # K + MASK token
        max_seq_len: int = 240,      # 80 patches x 3 levels
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = None,            # defaults to 4 * d_model
        dropout: float = 0.1,
        n_levels: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.real_vocab = vocab_size - 1  # K (excludes MASK)
        self.mask_token = vocab_size - 1
        self.max_seq_len = max_seq_len
        self.n_levels = n_levels
        if d_ff is None:
            d_ff = 4 * d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.level_emb = nn.Embedding(n_levels, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.real_vocab)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.level_emb.weight, std=0.02)

    def forward(self, x, t, padding_mask=None):
        """Forward pass.

        Args:
            x: (B, L) token IDs (may contain MASK token).
            t: (B,) continuous time in [0, 1].
            padding_mask: (B, L) bool, True = valid token.

        Returns:
            logits: (B, L, real_vocab) -- predictions for real tokens only.
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device)
        levels = positions % self.n_levels

        h = self.token_emb(x) + self.pos_emb(positions) + self.level_emb(levels)
        h = h + self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)  # broadcast time
        h = self.drop(h)

        # Transformer (no causal mask -- bidirectional)
        src_key_padding_mask = ~padding_mask if padding_mask is not None else None
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, n_samples=1, seq_len=None, n_steps=100, temperature=1.0):
        """Generate sequences via iterative unmasking.

        Starts from all-MASK, iteratively unmasks highest-confidence positions.
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        device = next(self.parameters()).device
        x = torch.full((n_samples, seq_len), self.mask_token, device=device, dtype=torch.long)
        padding_mask = torch.ones(n_samples, seq_len, device=device, dtype=torch.bool)

        for step in range(n_steps):
            t_val = 1.0 - step / n_steps
            t = torch.full((n_samples,), t_val, device=device)
            logits = self(x, t, padding_mask)
            probs = F.softmax(logits / temperature, dim=-1)

            # For each masked position, compute confidence (max prob)
            is_masked = (x == self.mask_token)
            confidence = probs.max(dim=-1).values  # (B, L)
            confidence[~is_masked] = -1.0  # ignore already-unmasked

            # Unmask top-k positions this step
            n_to_unmask = max(1, int(is_masked.float().sum(-1).max().item() / max(n_steps - step, 1)))
            for b in range(n_samples):
                masked_positions = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                conf_at_masked = confidence[b, masked_positions]
                topk = min(n_to_unmask, len(masked_positions))
                _, top_idx = conf_at_masked.topk(topk)
                for idx in top_idx:
                    pos = masked_positions[idx]
                    sampled = torch.multinomial(probs[b, pos], 1).item()
                    x[b, pos] = sampled

        # Replace any remaining MASK with random tokens
        still_masked = (x == self.mask_token)
        if still_masked.any():
            x[still_masked] = torch.randint(0, self.real_vocab, (still_masked.sum(),), device=device)

        return x


def apply_masking(tokens, t, padding_mask, mask_token=1024):
    """Apply continuous-time masking for MDLM training.

    Args:
        tokens: (B, L) original tokens.
        t: (B,) mask probability per sample.
        padding_mask: (B, L) True = valid.
        mask_token: ID for MASK token.

    Returns:
        masked_tokens: (B, L) with some positions replaced by mask_token.
        mask_positions: (B, L) bool, True = was masked.
    """
    B, L = tokens.shape
    rand = torch.rand(B, L, device=tokens.device)
    mask_prob = t.unsqueeze(-1).expand(B, L)
    mask_positions = (rand < mask_prob) & padding_mask
    masked_tokens = tokens.clone()
    masked_tokens[mask_positions] = mask_token
    return masked_tokens, mask_positions
