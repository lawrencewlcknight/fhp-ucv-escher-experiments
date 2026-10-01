"""Policy-only residual card architectures; no privileged/full-state inputs.

The exact canonical 183-feature pathway is never removed. The suit branches
share rank/role-aware weights across four suits and pool without suit labels.
"""

import numpy as np
import torch
from torch import nn

from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear
from .features import POLICY_LAYOUT, StructuredFHPMLP


MODEL_TYPE = "fhp_card_residual_v1"
KINDS = ("dense", "deepsets", "attention")


def suit_tokens(inputs):
    """[..., suit, hole-ranks + public-ranks]; ranks and card roles are ordered."""
    shape = (*inputs.shape[:-1], 4, 13)
    return torch.cat((inputs[..., :52].reshape(shape), inputs[..., 52:104].reshape(shape)), -1)


def invariant_context(inputs):
    # Exclude the eight suit-indexed count features (164:172). All other
    # context is suit-invariant, including player, round, exact betting,
    # hole/public rank counts, pocket flags and hand category: 71 dimensions.
    return torch.cat((inputs[..., 104:164], inputs[..., 172:183]), -1)


class SuitAttention(nn.Module):
    """One pre-normalised, four-head block; no positions, dropout or suit IDs."""

    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.qkv = SonnetLinear(64, 192, False)
        self.project = SonnetLinear(64, 64, False)
        self.feedforward = nn.Sequential(SonnetLinear(64, 128), SonnetLinear(128, 64, False))

    def forward(self, tokens):
        original_shape = tokens.shape
        value = tokens.reshape(-1, 4, 64)
        q, k, v = self.qkv(self.norm1(value)).reshape(-1, 4, 3, 4, 16).permute(2, 0, 3, 1, 4).unbind(0)
        attention = torch.softmax((q @ k.transpose(-1, -2)) * .25, -1)
        mixed = (attention @ v).transpose(1, 2).reshape(-1, 4, 64)
        value = value + self.project(mixed)
        value = value + self.feedforward(self.norm2(value))
        return value.reshape(original_shape)


class ResidualCardPolicy(nn.Module):
    """Trainable canonical baseline plus an initially zero logit correction."""

    def __init__(self, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unsupported residual card architecture: {kind}")
        self.kind = kind
        self.layout = POLICY_LAYOUT
        self.input_size, self.output_size = 183, 3
        self.hidden_layers = [192, 192]
        # Construct first: a common initialization seed gives every arm the
        # identical baseline parameters, regardless of the residual branch.
        self.base = StructuredFHPMLP(POLICY_LAYOUT, self.hidden_layers, 3, branch_width=64)
        if kind == "dense":
            self.residual = nn.Sequential(SonnetLinear(183, 200), SonnetLinear(200, 110),
                                          ZeroInitLinear(110, 3, False))
        else:
            self.suit_encoder = nn.Sequential(SonnetLinear(26, 64), SonnetLinear(64, 64))
            self.attention = SuitAttention() if kind == "attention" else nn.Identity()
            if kind == "deepsets":
                self.residual = nn.Sequential(SonnetLinear(199, 184), SonnetLinear(184, 88),
                                              ZeroInitLinear(88, 3, False))
            else:
                self.residual = nn.Sequential(SonnetLinear(199, 96), ZeroInitLinear(96, 3, False))

    def residual_logits(self, inputs):
        if inputs.shape[-1] != self.input_size:
            raise ValueError("Card policy requires the exact 183-feature information state")
        if self.kind == "dense":
            return self.residual(inputs)
        tokens = self.attention(self.suit_encoder(suit_tokens(inputs)))
        pooled = torch.cat((tokens.mean(-2), tokens.amax(-2), invariant_context(inputs)), -1)
        return self.residual(pooled)

    def forward(self, inputs):
        return self.base(inputs) + self.residual_logits(inputs)

    def checkpoint_metadata(self):
        return {"type": MODEL_TYPE, "version": 1, "kind": self.kind,
                "layout": POLICY_LAYOUT.to_dict(), "base": self.base.checkpoint_metadata(),
                "pooling": None if self.kind == "dense" else "mean_and_max",
                "suit_width": None if self.kind == "dense" else 64,
                "suit_encoder_layers": None if self.kind == "dense" else [64, 64],
                "residual_hidden_layers": {"dense": [200, 110], "deepsets": [184, 88], "attention": [96]}[self.kind],
                "attention_heads": 4 if self.kind == "attention" else 0,
                "attention_feedforward_width": 128 if self.kind == "attention" else None,
                "output_size": 3}


def new_model(settings, seed):
    """Initialize without changing sampler, NumPy or Torch random state."""
    if settings.get("model_type") != MODEL_TYPE:
        raise ValueError("Unsupported card model specification")
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return ResidualCardPolicy(settings["kind"])
    finally:
        np.random.set_state(numpy_state)
