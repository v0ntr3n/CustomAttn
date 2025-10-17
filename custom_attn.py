# ---- FIXED custom cross-attention (QK-Norm + text-gated), version-robust ----
import torch, torch.nn as nn, torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0 as SDPProc

# Prefer Flash/MemEff kernels if available
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    pass

class QKNormGatedCrossAttn(nn.Module):
    def __init__(self, qk_norm: bool=True, gate_heads: bool=True):
        super().__init__()
        self.qk_norm, self.gate_heads = qk_norm, gate_heads
        self._built = False
        self.context_proj = None
        self.head_gain = None

    @staticmethod
    def _num_heads(attn):
        return getattr(attn, "heads", getattr(attn, "num_heads", None))

    @staticmethod
    def _head_dim(attn):
        # diffusers≥0.27 usually has dim_head; older forks used head_dim
        if hasattr(attn, "head_dim"):  # rare
            return attn.head_dim
        if hasattr(attn, "dim_head"):
            return attn.dim_head
        # fallback: infer from to_q out_features
        return attn.to_q.out_features // QKNormGatedCrossAttn._num_heads(attn)

    @torch.no_grad()
    def _l2(self, x, dim=-1, eps=1e-6):
        return F.normalize(x, p=2, dim=dim, eps=eps)

    def _lazy_build(self, encoder_hidden_states, n_heads, device, dtype):
        if self._built: return
        ctx_dim = encoder_hidden_states.shape[-1] if encoder_hidden_states is not None else 768
        if self.gate_heads:
            self.context_proj = nn.Linear(ctx_dim, n_heads, bias=True).to(device=device, dtype=dtype)
            self.head_gain = nn.Parameter(torch.zeros(n_heads, device=device, dtype=dtype))
        self._built = True

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        bsz, q_len, _ = hidden_states.shape
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Projections
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        n_heads = self._num_heads(attn)
        d_head  = self._head_dim(attn)

        # [B, L, C] -> [B*H, L, Dh]
        def pack(x):
            return x.view(bsz, -1, n_heads, d_head).transpose(1, 2).reshape(bsz * n_heads, -1, d_head)
        q, k, v = pack(q), pack(k), pack(v)

        # QK-Norm (cosine attention)
        if self.qk_norm:
            q = self._l2(q); k = self._l2(k)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p=0.0, is_causal=False)
        out = out.reshape(bsz, n_heads, q_len, d_head).transpose(1, 2).reshape(bsz, q_len, n_heads * d_head)

        # Text-conditioned per-head gating
        if encoder_hidden_states is not None:
            self._lazy_build(encoder_hidden_states, n_heads, out.device, out.dtype)
            if self.gate_heads:
                pooled = encoder_hidden_states.mean(dim=1)            # [B, C]
                gamma  = torch.sigmoid(self.context_proj(pooled))     # [B, H]
                gamma  = gamma * torch.sigmoid(self.head_gain)        # [H]
                gamma  = gamma.view(bsz, 1, n_heads).repeat(1, q_len, 1)
                out = out.view(bsz, q_len, n_heads, d_head)
                out = out * (1.0 + gamma.unsqueeze(-1))
                out = out.view(bsz, q_len, n_heads * d_head)

        # Standard to_out: linear then dropout
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

# Attach custom to cross-attn only; keep self-attn on fast SDP
procs, num_cross = {}, 0
for name in pipe.unet.attn_processors.keys():
    if name.endswith("attn2.processor"):      # cross-attention
        procs[name] = QKNormGatedCrossAttn()
        num_cross += 1
    else:                                     # self-attention
        procs[name] = SDPProc()
pipe.unet.set_attn_processor(procs)
print(f"[CustomAttn] attached to {num_cross} cross-attention sites.")
