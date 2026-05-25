import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from timm.layers import DiffAttention


def _extract_attn(model, x):
    records = []
    hooks = []

    for block in model.blocks:
        rec = {}
        records.append(rec)

        def make_hook(attn_mod, rec):
            def hook(mod, inp, out):
                x = inp[0].detach()
                B, N, _ = x.shape
                with torch.no_grad():
                    if isinstance(attn_mod, DiffAttention):
                        nh, hd = attn_mod.num_heads, attn_mod.head_dim
                        q, k, v = attn_mod.qkv(x).chunk(3, dim=2)
                        q = q.reshape(B, N, 2*nh, hd).transpose(1, 2)
                        k = k.reshape(B, N, 2*nh, hd).transpose(1, 2)
                        q = q.reshape(B, nh, 2, N, hd)
                        k = k.reshape(B, nh, 2, N, hd)
                        q1, q2 = q.unbind(2)
                        k1, k2 = k.unbind(2)
                        scale = hd ** -0.5
                        attn1 = (q1 * scale @ k1.transpose(-2, -1)).softmax(-1)
                        attn2 = (q2 * scale @ k2.transpose(-2, -1)).softmax(-1)
                        rec.update(dict(is_diff=True, nh=nh, attn1=attn1, attn2=attn2))
                    else:
                        nh = attn_mod.num_heads
                        hd = attn_mod.qkv.in_features // nh
                        qkv = attn_mod.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
                        q, k, _ = qkv.unbind(0)
                        scale = hd ** -0.5
                        attn = (q * scale @ k.transpose(-2, -1)).softmax(-1)
                        rec.update(dict(is_diff=False, nh=nh, attn=attn))
            return hook

        hooks.append(block.attn.register_forward_hook(make_hook(block.attn, rec)))

    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()

    return records


def log_attn_maps(model, x, logger):
    sample = x[:1]
    N = model.patch_embed.num_patches
    grid_size = int(math.sqrt(N))

    records = _extract_attn(model, sample)

    # 열 수: 1(원본) + max(nh * attn_per_head)
    max_attn_cols = max((r['nh'] * (2 if r['is_diff'] else 1)) for r in records)
    ncols = 1 + max_attn_cols
    nrows = len(records)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6))

    # 원본 이미지 (denormalize 없이 clip만)
    img = sample[0].cpu().float().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    for row, rec in enumerate(records):
        ax_row = axes[row]
        is_diff = rec['is_diff']
        nh = rec['nh']

        # 첫 열: 원본 이미지
        ax_row[0].imshow(img)
        ax_row[0].axis('off')
        if row == 0:
            ax_row[0].set_title('Image', fontsize=7)
        label = f"L{row}\n({'diff' if is_diff else 'std'})"
        ax_row[0].set_ylabel(label, fontsize=6, rotation=0, labelpad=30, va='center')

        # 나머지 열: head별 CLS attn
        col = 1
        for h in range(nh):
            if is_diff:
                maps = [rec['attn1'][0, h, 0, 1:], rec['attn2'][0, h, 0, 1:]]
                titles = [f'H{h} attn1', f'H{h} attn2']
            else:
                maps = [rec['attn'][0, h, 0, 1:]]
                titles = [f'H{h}']

            for m, title in zip(maps, titles):
                grid = m.cpu().float().reshape(grid_size, grid_size).numpy()
                ax_row[col].imshow(grid, cmap='viridis')
                ax_row[col].axis('off')
                if row == 0:
                    ax_row[col].set_title(title, fontsize=7)
                col += 1

        # 남은 열 비우기
        for c in range(col, ncols):
            ax_row[c].axis('off')

    plt.tight_layout()

    if logger is not None:
        logger.experiment.log({'attn/cls_attention': wandb.Image(fig)})
    plt.close(fig)
