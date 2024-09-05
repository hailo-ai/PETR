import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
Tensor = torch.Tensor

def _split_scaled_dot_product_attention(
        query_split: list,
        key_split: list,
        value_split: list,
        num_heads: int, 
        num_split: int, 
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    tgt_len, bsz, embed_dim = query_split[0].size()
    assert num_split * (num_heads // num_split) == num_heads, f"num_heads {num_heads} not divisible by num_split {num_split}"
    head_dim = embed_dim // (num_heads // num_split)
    assert head_dim * (num_heads // num_split) == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads//num_splits {num_heads//num_split}"

    if attn_mask is not None:
        attn_mask_split = attn_mask.chunk(num_split)
    attn = []
    output = []
    for i in range(num_split):
        query_split[i] = query_split[i].view(tgt_len, bsz * num_heads // num_split , head_dim).transpose(0, 1)
        query_split[i] = query_split[i] / math.sqrt(query_split[i].shape[-1])
        key_split[i]   = key_split[i].view(-1, bsz * num_heads // num_split, head_dim).transpose(0, 1)
        value_split[i] = value_split[i].view(-1, bsz * num_heads // num_split, head_dim).transpose(0, 1)
        attn_split = torch.bmm(query_split[i], key_split[i].transpose(-2, -1))
        if attn_mask is not None:
            attn_split += attn_mask_split[i]
        attn_split = F.softmax(attn_split, dim=-1)
        if dropout_p > 0.0:
            attn_split = F.dropout(attn_split, p=dropout_p)
        output_split = torch.bmm(attn_split, value_split[i])
        attn.append(attn_split)
        output.append(output_split)

    attn_output = torch.cat(output, dim=0)
    attn_output_weights = torch.cat(attn, dim=0)

    return attn_output, attn_output_weights

class SplitPETRMultiheadAttention(nn.MultiheadAttention):
    
    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.,
                 num_head_split=1,
                 **kwargs):
        super(SplitPETRMultiheadAttention, self).__init__(embed_dims,
                                                          num_heads,
                                                          dropout)
        self.num_split = num_head_split
    
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        
        # input projection
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        q_split = list(q.chunk(self.num_split, dim=-1))
        k_split = list(k.chunk(self.num_split, dim=-1))
        v_split = list(v.chunk(self.num_split, dim=-1))

        # update source sequence length after adjustments
        src_len = k.view(-1, bsz * num_heads, head_dim).transpose(0, 1).size(1)
        
        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
        
        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        # adjust dropout probability
        if not self.training:
            self.dropout = 0.0

        attn_output, attn_output_weights = _split_scaled_dot_product_attention(q_split, k_split, v_split, num_heads, self.num_split, attn_mask, self.dropout)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None
