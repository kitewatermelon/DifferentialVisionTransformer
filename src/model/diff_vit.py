import timm
import torch
import torch.nn as nn
from timm.layers import DiffAttention

from typing import Container

def replace_attention_with_diff(
    model: nn.Module,
    replace_indices: Container[int] | None = None,
    start_idx: int | None = None,
) -> nn.Module:
    """
    replace_indices: {6, 7, 8, 9, 10, 11} 같이 명시적 지정
    start_idx:       6 이상 전부 교체하는 단순 케이스
    둘 다 None이면 전체 교체
    """
    total = len(model.blocks)
    
    if replace_indices is None:
        if start_idx is not None:
            replace_indices = set(range(start_idx, total))
        else:
            replace_indices = set(range(total))  # 전부
    
    for depth_idx, block in enumerate(model.blocks):
        if depth_idx not in replace_indices:
            continue
        
        old_attn = block.attn
        dim = old_attn.proj.in_features
        num_heads = old_attn.num_heads
        
        new_attn = DiffAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=old_attn.qkv.bias is not None,
            proj_bias=old_attn.proj.bias is not None,
            attn_drop=old_attn.attn_drop.p,
            proj_drop=old_attn.proj_drop.p,
            depth=depth_idx,
        )
        
        block.attn = new_attn
    
    return model

def get_diff_vit(name='vit_tiny_patch16_224'):
    model = timm.create_model(name, pretrained=False)
    model = replace_attention_with_diff(model=model, start_idx=0)
    return model

def get_a6d6(name='vit_tiny_patch16_224'):
    model = timm.create_model(name, pretrained=False)
    model = replace_attention_with_diff(model=model, start_idx=6)
    return model

def get_odd_diff_vit(name='vit_tiny_patch16_224'):
    model = timm.create_model(name, pretrained=False)
    model = replace_attention_with_diff(model, replace_indices={0, 2, 4, 6, 8, 10})
    return model

def get_even_diff_vit(name='vit_tiny_patch16_224'):
    model = timm.create_model(name, pretrained=False)
    model = replace_attention_with_diff(model, replace_indices={1, 3, 5, 7, 9, 11})
    return model

if __name__=="__main__":
    # print(get_diff_vit(name='vit_tiny_patch16_224'))
    print(get_odd_diff_vit(name='vit_tiny_patch16_224'))