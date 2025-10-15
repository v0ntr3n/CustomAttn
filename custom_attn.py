
import torch
import torch.nn as nn
import torch.nn.functional as F
class QKNormGatedCrossAttn(nn.Module):
    def __init__(self, heads: int = None, qk_norm: bool = True, gate_heads: bool = True):
        super().__init__()
        self.qk_norm = qk_norm
        self.gate_heads = gate_heads
        self.heads = heads
        self._built = False
        self.context_proj = None
        self.head_gain = None
    def _lazy_build(self, encoder_hidden_states, num_heads, device, dtype):
        if self._built:
            return
        ctx_dim = encoder_hidden_states.shape[-1] if encoder_hidden_states is not None else 768
        h = num_heads if self.heads is None else self.heads
        if self.gate_heads:
            self.context_proj = nn.Linear(ctx_dim, h, bias=True).to(device=device, dtype=dtype)
            self.head_gain = nn.Parameter(torch.zeros(h, device=device, dtype=dtype))
        self.heads = h
        self._built = True
    @torch.no_grad()
    def _l2_normalize(self, x, dim=-1, eps=1e-6):
        return F.normalize(x, p=2, dim=dim, eps=eps)
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        bsz, q_len, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        num_heads = attn.heads
        head_dim  = attn.head_dim
        def H(x):
            x = x.view(bsz, -1, num_heads, head_dim).transpose(1, 2).reshape(bsz * num_heads, -1, head_dim)
            return x
        q, k, v = H(query), H(key), H(value)
        if self.qk_norm:
            q = self._l2_normalize(q, dim=-1)
            k = self._l2_normalize(k, dim=-1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p=0.0, is_causal=False)
        out = out.reshape(bsz, num_heads, q_len, head_dim).transpose(1, 2).reshape(bsz, q_len, num_heads * head_dim)
        if encoder_hidden_states is not None:
            self._lazy_build(encoder_hidden_states, num_heads, out.device, out.dtype)
            if self.gate_heads:
                pooled = encoder_hidden_states.mean(dim=1)
                gamma  = torch.sigmoid(self.context_proj(pooled))
                gamma  = gamma * torch.sigmoid(self.head_gain)
                gamma  = gamma.view(bsz, 1, num_heads).repeat(1, q_len, 1)
                out = out.view(bsz, q_len, num_heads, head_dim)
                out = out * (1.0 + gamma.unsqueeze(-1))
                out = out.view(bsz, q_len, num_heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out, scale=attn.rescale_output_factor) if isinstance(attn.to_out, nn.ModuleList) else out
        return out
