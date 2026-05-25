import timm
import torch.nn as nn

def get_vit(name: str ='vit_tiny_patch16_224')-> nn.Module:
    model = timm.create_model(name, pretrained=False)
    return model