# Modified by Aether AI for CausalWM (2026). Original file: Lightricks LTX-2 (`ltx-core`).
# Distributed under the LTX-2 Community License Agreement; see LICENSE and NOTICE.

from typing import Optional, Tuple

import torch

from ltx_core.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings

# Number of AdaLN modulation parameters per transformer block.
# Base: 2 params (shift + scale) x 3 norms (self-attn, feed-forward, output).
ADALN_NUM_BASE_PARAMS = 6
# Cross-attention AdaLN adds 3 more (scale, shift, gate) for the CA norm.
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    """Total number of AdaLN parameters per block."""
    return ADALN_NUM_BASE_PARAMS + (ADALN_NUM_CROSS_ATTN_PARAMS if cross_attention_adaln else 0)


class _ActionMlp(torch.nn.Module):
    """2-layer MLP (Linear -> GELU(tanh) -> Linear). One head of the dual-path ActionEmbedder."""

    _FC2_INIT_STD = 1e-3

    def __init__(self, action_in: int, out_dim: int, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(action_in, hidden)
        self.act = torch.nn.GELU(approximate="tanh")
        self.fc2 = torch.nn.Linear(hidden, out_dim)
        torch.nn.init.normal_(self.fc2.weight, std=self._FC2_INIT_STD)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(action)))


class ActionEmbedder(torch.nn.Module):
    r"""Dual-path action embedder.

    The action is injected through two parallel MLPs:
      * ``head_main``: ``action_in -> embedding_dim`` (D). Added to
        ``embedded_timestep`` inside ``AdaLayerNormSingle`` (main path; goes
        through the AdaLN modulation linear like the timestep).
      * ``head_mod``:  ``action_in -> modulation_dim`` (= coeff * embedding_dim,
        the width of the AdaLN modulation tensor). Added directly to the
        per-token modulation (``TransformerArgs.timesteps``) before
        ``get_ada_values``, i.e. straight onto every block's scale/shift/gate,
        bypassing the modulation linear.

    ``forward`` returns ``(main_emb (..., D), mod_emb (..., modulation_dim))``.

    ``modulation_dim = None`` falls back to single-path (head_mod disabled, only
    the main path).
    """

    def __init__(
        self,
        action_in: int,
        embedding_dim: int,
        modulation_dim: int | None = None,
        hidden_mult: int = 4,
    ):
        super().__init__()
        hidden = embedding_dim * hidden_mult
        self.head_main = _ActionMlp(action_in, embedding_dim, hidden)
        self.head_mod = _ActionMlp(action_in, modulation_dim, hidden) if modulation_dim else None

    def forward(self, action: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        main_emb = self.head_main(action)
        mod_emb = self.head_mod(action) if self.head_mod is not None else None
        return main_emb, mod_emb


class ActionInputProjection(torch.nn.Module):
    r"""Per-pair MLP projecting a raw robot action vector to the
    latent-action width.

    Default structure:
    ``Linear(di,256) -> GELU -> Linear(256,256) -> GELU -> Linear(256,do)``.
    The output width matches ``latent_action_dim`` so the downstream
    :class:`ActionEmbedder` (input width ``latent_action_dim * npf``) is reused
    unchanged.

    Applied per action pair (last dim = ``action_in``) before the embedder
    (see ``TransformerArgsPreprocessor._prepare_action``).
    """

    def __init__(
        self,
        action_in: int,
        out_dim: int,
        hidden: int = 256,
        net_spec: dict | None = None,
    ):
        super().__init__()
        self.action_in = action_in
        self.out_dim = out_dim
        if net_spec is None:
            # Default structure:
            #   Linear(di,256) -> GELU -> Linear(256,256) -> GELU -> Linear(256,do)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(action_in, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, out_dim),
            )
        else:
            # net_spec: a depth-N block stack with optional LayerNorm + Dropout, then
            # a linear head.
            spec_hidden = int(net_spec.get("hidden", hidden))
            depth = int(net_spec.get("depth", 2))
            dropout = float(net_spec.get("dropout", 0.0))
            norm = net_spec.get("norm", "none")
            # Dropout slots use p=0; they only keep the Sequential indices (and thus
            # the state-dict keys) aligned with the spec.
            keep_dropout_slot = dropout > 0
            layers: list[torch.nn.Module] = []
            d = action_in
            for _ in range(depth):
                layers.append(torch.nn.Linear(d, spec_hidden))
                if norm == "layer":
                    layers.append(torch.nn.LayerNorm(spec_hidden))
                layers.append(torch.nn.GELU())
                if keep_dropout_slot:
                    layers.append(torch.nn.Dropout(0.0))  # disabled; slot kept for key alignment
                d = spec_hidden
            layers.append(torch.nn.Linear(d, out_dim))
            self.net = torch.nn.Sequential(*layers)
        # Default parameter init (overwritten by a checkpoint load).
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                torch.nn.init.zeros_(m.bias)

        # Optional input/output normalization (off by default). When enabled via
        # ``set_bridge_stats``, forward becomes
        #   z = g((a - action_mean) / action_std) * zsd + zm
        # i.e. ``net`` (g) maps a z-scored raw action to a normalized latent action,
        # which is then de-normalized. The four stats are fixed (non-persistent)
        # buffers.
        self.use_bridge_protocol = False

    def set_bridge_stats(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        zm: torch.Tensor,
        zsd: torch.Tensor,
    ) -> None:
        """Enable input/output normalization with fixed affine stats.

        ``action_mean``/``action_std`` are the per-dim (action_in) stats of the
        raw action; ``zm``/``zsd`` the per-dim (out_dim) stats of the latent
        action. Dead dims (std<1e-6) use std=1. Registered as non-persistent
        float32 buffers.
        """
        am = torch.as_tensor(action_mean, dtype=torch.float32).reshape(-1)
        asd = torch.as_tensor(action_std, dtype=torch.float32).reshape(-1)
        zmv = torch.as_tensor(zm, dtype=torch.float32).reshape(-1)
        zsdv = torch.as_tensor(zsd, dtype=torch.float32).reshape(-1)
        if am.numel() != self.action_in or asd.numel() != self.action_in:
            raise ValueError(
                f"bridge action stats dim {am.numel()}/{asd.numel()} != action_in {self.action_in}"
            )
        if zmv.numel() != self.out_dim or zsdv.numel() != self.out_dim:
            raise ValueError(f"bridge z stats dim {zmv.numel()}/{zsdv.numel()} != out_dim {self.out_dim}")
        asd = torch.where(asd < 1e-6, torch.ones_like(asd), asd)
        zsdv = torch.where(zsdv < 1e-6, torch.ones_like(zsdv), zsdv)
        self.register_buffer("action_mean", am, persistent=False)
        self.register_buffer("action_std", asd, persistent=False)
        self.register_buffer("zm", zmv, persistent=False)
        self.register_buffer("zsd", zsdv, persistent=False)
        self.use_bridge_protocol = True

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """``action`` (..., action_in) -> (..., out_dim). Acts on the last dim."""
        if not self.use_bridge_protocol:
            return self.net(action)
        # z-score the raw action and de-normalize the output in float32 (robust to
        # tiny per-dim std); run ``net`` in the action's dtype to match its params.
        an = (action.to(torch.float32) - self.action_mean) / self.action_std
        zn = self.net(an.to(action.dtype))
        z = zn.to(torch.float32) * self.zsd + self.zm
        return z.to(action.dtype)


class AdaLayerNormSingle(torch.nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).
    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
        action_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        if action_embed is not None:
            # Action injection: add the action embedding to the
            # timestep embedding BEFORE the modulation linear, so it propagates
            # into every scale/shift (and gate) the linear produces. This is the
            # single numerical injection point. action_embed must already be
            # broadcast to embedded_timestep's per-token layout (done in the
            # preprocessor). action_embed=None → bit-equivalent to base model.
            embedded_timestep = embedded_timestep + action_embed.to(embedded_timestep.dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
