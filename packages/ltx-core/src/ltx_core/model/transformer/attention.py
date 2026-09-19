# Modified by Aether AI for CausalWM (2026). Original file: Lightricks LTX-2 (`ltx-core`).
# Distributed under the LTX-2 Community License Agreement; see LICENSE and NOTICE.

from enum import Enum
from typing import Protocol

import torch

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            if mask.dtype == torch.bool:
                # SDPA bool True means visible; xformers expects additive bias.
                bias = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float("-inf"))
            else:
                bias = mask.to(q.dtype)
            mask_out[..., : mask.shape[-1]] = bias
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class CudnnAttention(AttentionCallable):
    """``cudnn`` config alias for mask-capable PyTorch SDPA dispatch.

    Preserve the runtime's default priority (Flash, Efficient, Math, cuDNN on
    PyTorch 2.9). Listing cuDNN first does NOT prioritize it.
    Efficient is essential for arbitrary padding masks rejected by Flash;
    otherwise Math materializes the quadratic attention matrix. This alias
    permits cuDNN but deliberately does not promise that cuDNN is selected.
    """

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

        # MATH is also required when another Python thread executes an FP32
        # SDPA call while this context is active. PyTorch's
        # SDPA backend selection is process-global, so omitting MATH can make
        # an otherwise independent FP32 attention call fail with "No available
        # kernel". Retain the default priority, including the no-mask Flash path.
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
             SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    CUDNN = "cudnn"
    DEFAULT = "default"

    def to_callable(self) -> AttentionCallable:
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks.

        DEFAULT path honors the LTX2_ATTENTION_BACKEND env var (cudnn/xformers/
        pytorch/flash_attention_3); unset falls through to the auto behavior
        (xformers if installed else pytorch), so backends can be compared
        without touching calling sites.
        """
        import os as _os  # noqa: PLC0415

        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()
        elif self is AttentionFunction.CUDNN:
            return CudnnAttention()
        else:  # DEFAULT
            env_choice = _os.environ.get("LTX2_ATTENTION_BACKEND", "").strip().lower()
            if env_choice == "cudnn":
                return CudnnAttention()
            elif env_choice == "xformers":
                return XFormersAttention()
            elif env_choice == "pytorch":
                return PytorchAttention()
            elif env_choice == "flash_attention_3":
                return FlashAttention3()
            # Legacy auto behavior: XFormers if installed else - PyTorch
            return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = (
            attention_function.to_callable()
            if isinstance(attention_function, AttentionFunction)
            else attention_function
        )

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def _strict_stage_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stream_slices: tuple[int, ...],
        stream_ids: tuple[int, ...] | None,
        stream_obs_len: int,
        stream_stages: tuple[int, ...],
        invalid_key_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Strict, stage-ordered block attention (multi-pass SDPA).

        Layout ``[video (n0) | stream_1 | stream_2 | ...]``; video = obs prefix
        ``[0:nob]`` (sigma == 0) + future ``[nob:n0]``.

            obs q      -> obs kv only
            future q   -> all video kv (+ tanh-gated cross into every stream)
            stream_i q -> obs kv + streams with stage < stage_i + streams with stage == stage_i

        Never: obs -> future/streams (blocks two-hop leakage); stream -> future
        video; stream -> higher stage. ``invalid_key_mask`` drops invalid tokens
        from every key set (boolean SDPA mask on the gathered key subset).
        """
        n0 = stream_slices[0]
        nob = min(stream_obs_len, n0)
        gates = getattr(self, "icl_stream_gates", None)
        segs: list[tuple[int, int]] = []
        off = n0
        for ln in stream_slices[1:]:
            segs.append((off, ln))
            off += ln
        if off != q.shape[1]:
            raise ValueError(f"stream_slices {stream_slices} do not cover seq len {q.shape[1]}")

        def _attend(qs: torch.Tensor, key_ranges: list[tuple[int, int]]) -> torch.Tensor:
            ks = torch.cat([k[:, a : a + n] for a, n in key_ranges], dim=1)
            vs = torch.cat([v[:, a : a + n] for a, n in key_ranges], dim=1)
            mask = None
            if invalid_key_mask is not None:
                inv = torch.cat([invalid_key_mask[:, a : a + n] for a, n in key_ranges], dim=1)  # (B, K)
                if bool(inv.any()):
                    # bool mask, True = attend; expand over queries (B, Q, K)
                    mask = (~inv).unsqueeze(1).expand(-1, qs.shape[1], -1)
            return self.attention_function(qs, ks, vs, self.heads, mask)

        outs = []
        if nob > 0:
            outs.append(_attend(q[:, :nob], [(0, nob)]))
        if nob < n0:
            out_fut = _attend(q[:, nob:n0], [(0, n0)])
            if gates is not None:
                for i, (a, ln) in enumerate(segs):
                    sid = stream_ids[i] if stream_ids is not None else i
                    cross = _attend(q[:, nob:n0], [(a, ln)])
                    out_fut = out_fut + torch.tanh(gates[sid]).to(out_fut.dtype) * cross
            outs.append(out_fut)
        for i, (a, ln) in enumerate(segs):
            ranges = [(0, nob)] if nob > 0 else []
            ranges += [segs[j] for j in range(len(segs)) if j != i and stream_stages[j] <= stream_stages[i]]
            ranges.append((a, ln))
            outs.append(_attend(q[:, a : a + ln], ranges))
        return torch.cat(outs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        stream_slices: tuple[int, ...] | None = None,
        stream_self_only: bool = False,
        stream_ids: tuple[int, ...] | None = None,
        stream_obs_len: int | None = None,
        stream_stages: tuple[int, ...] | None = None,
        invalid_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.

        ``stream_stages``: one stage id per aux stream (aligned with
        ``stream_slices[1:]``). Requires the strict layout (``stream_obs_len``).
        Stream i queries then read obs video kv + every stream with a lower
        stage + every stream with the SAME stage (itself included), so streams
        of one stage are generated jointly. ``None`` keeps the ``chain``
        rule. ``invalid_key_mask`` ``(B, T)`` bool (True = token is
        invalid) removes those tokens from the KEY set of every strict pass;
        obs tokens are never masked (frame 0 is always valid).
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA).
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            if stream_slices is None:
                out = self.attention_function(q, k, v, self.heads, mask)  # (B, T, H*D)
            else:
                # Auxiliary streams: asymmetric block attention as multi-pass SDPA
                # over sliced q/k/v instead of a dense mask (keeps fast unmasked
                # kernels; avoids exotic masked-kernel paths). Layout along the
                # sequence: [video (n0), stream1 (n1), stream2 (n2), ...].
                #   video queries  -> video kv via the ORIGINAL self-attn pass
                #     (bitwise-identical to the no-aux forward), plus — when
                #     ``self.icl_stream_gates`` exists — a per-stream tanh-gated
                #     cross-attention term into each stream's kv.
                #   stream_i queries -> video kv (+ EARLIER streams' kv when
                #     ``self.icl_stream_chain``: chain visibility, sequence order
                #     = generation order) + own kv.
                if mask is not None:
                    raise ValueError("stream_slices does not compose with an explicit attention mask")
                n0 = stream_slices[0]
                gates = getattr(self, "icl_stream_gates", None)
                chain = bool(getattr(self, "icl_stream_chain", False))
                if stream_stages is not None and (stream_obs_len is None or stream_self_only):
                    raise ValueError("stream_stages requires the strict layout (stream_obs_len) and not stream_self_only")
                if stream_stages is not None and len(stream_stages) != len(stream_slices) - 1:
                    raise ValueError(f"stream_stages {stream_stages} vs {len(stream_slices) - 1} aux slices")
                if invalid_key_mask is not None and stream_stages is None:
                    raise ValueError("invalid_key_mask is only implemented for the stream_stages path")
                if stream_stages is not None:
                    out = self._strict_stage_attention(
                        q, k, v, stream_slices, stream_ids, stream_obs_len, stream_stages, invalid_key_mask
                    )
                elif stream_obs_len is not None and not stream_self_only:
                    # Strict CoT mask. Video splits into obs [0:nob] (sigma==0
                    # observed frames, always a prefix) and future [nob:n0]
                    # (the "answer"):
                    #   obs q    -> obs kv ONLY (stays a pure observation
                    #               encoding — blocks second-order leakage);
                    #   future q -> all video kv + tanh-gated cross into every
                    #               stream (the answer reads all thoughts);
                    #   stream_i q -> obs + chain-EARLIER streams + self
                    #               (never future video: shortcut-proof).
                    nob = min(stream_obs_len, n0)
                    outs = []
                    # zero-length slices crash fused CUDA kernels ("No available
                    # kernel ... zero seq_len_q") — guard both partitions.
                    # nob == n0 (all-clean RGB) correctly skips the
                    # future pass AND the gated feature cross entirely.
                    if nob > 0:
                        outs.append(self.attention_function(q[:, :nob], k[:, :nob], v[:, :nob], self.heads, None))
                    if nob < n0:
                        out_fut = self.attention_function(q[:, nob:n0], k[:, :n0], v[:, :n0], self.heads, None)
                        if gates is not None:
                            off = n0
                            for i, ln in enumerate(stream_slices[1:]):
                                sid = stream_ids[i] if stream_ids is not None else i
                                cross = self.attention_function(
                                    q[:, nob:n0], k[:, off : off + ln], v[:, off : off + ln], self.heads, None
                                )
                                out_fut = out_fut + torch.tanh(gates[sid]).to(out_fut.dtype) * cross
                                off += ln
                        outs.append(out_fut)
                    off = n0
                    prev_strict: list[tuple[int, int]] = []
                    for ln in stream_slices[1:]:
                        parts_k = [k[:, :nob]]
                        parts_v = [v[:, :nob]]
                        if chain:
                            for ps, pl in prev_strict:
                                parts_k.append(k[:, ps : ps + pl])
                                parts_v.append(v[:, ps : ps + pl])
                        parts_k.append(k[:, off : off + ln])
                        parts_v.append(v[:, off : off + ln])
                        outs.append(self.attention_function(
                            q[:, off : off + ln], torch.cat(parts_k, dim=1),
                            torch.cat(parts_v, dim=1), self.heads, None))
                        prev_strict.append((off, ln))
                        off += ln
                    if off != q.shape[1]:
                        raise ValueError(f"stream_slices {stream_slices} do not cover seq len {q.shape[1]}")
                    out = torch.cat(outs, dim=1)
                else:
                    # Non-strict path: video q -> all video kv (+ gated cross)
                    out0 = self.attention_function(q[:, :n0], k[:, :n0], v[:, :n0], self.heads, None)
                    if gates is not None and not stream_self_only:
                        off = n0
                        for i, ln in enumerate(stream_slices[1:]):
                            # stream_ids maps slice position -> stream identity (gate row)
                            sid = stream_ids[i] if stream_ids is not None else i
                            cross = self.attention_function(
                                q[:, :n0], k[:, off : off + ln], v[:, off : off + ln], self.heads, None
                            )
                            out0 = out0 + torch.tanh(gates[sid]).to(out0.dtype) * cross
                            off += ln
                    outs = [out0]
                    off = n0
                    prev_segs: list[tuple[int, int]] = []
                    for ln in stream_slices[1:]:
                        if stream_self_only:
                            # isolation mode: aux stream sees ONLY itself (no video kv)
                            ks, vs = k[:, off : off + ln], v[:, off : off + ln]
                        else:
                            parts_k = [k[:, :n0]]
                            parts_v = [v[:, :n0]]
                            if chain:
                                for ps, pl in prev_segs:
                                    parts_k.append(k[:, ps : ps + pl])
                                    parts_v.append(v[:, ps : ps + pl])
                            parts_k.append(k[:, off : off + ln])
                            parts_v.append(v[:, off : off + ln])
                            ks = torch.cat(parts_k, dim=1)
                            vs = torch.cat(parts_v, dim=1)
                        outs.append(self.attention_function(q[:, off : off + ln], ks, vs, self.heads, None))
                        prev_segs.append((off, ln))
                        off += ln
                    if off != q.shape[1]:
                        raise ValueError(f"stream_slices {stream_slices} do not cover seq len {q.shape[1]}")
                    out = torch.cat(outs, dim=1)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            # Reshape to (B, T, H, D) for per-head gating
            out = out.view(b, t, self.heads, self.dim_head)
            # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
            out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
            # Reshape back to (B, T, H*D)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)
