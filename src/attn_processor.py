import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class CustomFluxAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, height=44, width=88, attn_enforce=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.height = height
        self.width = width
        self.num_pixels = height * width
        self.attn_enforce = attn_enforce

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if self.attn_enforce != 1.0:
            attn_scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * attn.scale
            img_attn_scores = attn_scores[:, :, -self.num_pixels :, -self.num_pixels :]
            img_attn_scores = img_attn_scores.reshape(
                (batch_size, attn.heads, self.height, self.width, self.height, self.width)
            )
            img_attn_scores[:, :, :, self.width // 2 :, :, : self.width // 2] *= self.attn_enforce
            img_attn_scores = img_attn_scores.reshape(
                (batch_size, attn.heads, self.num_pixels, self.num_pixels)
            )
            attn_scores[:, :, -self.num_pixels :, -self.num_pixels :] = img_attn_scores
            attn_probs = attn_scores.softmax(dim=-1)
            hidden_states = torch.einsum("bhqk,bhkd->bhqd", attn_probs, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        return hidden_states
