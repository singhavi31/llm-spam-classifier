import torch
import torch.nn as nn
from ..config.config import BASE_CONFIG


def prepare_model_for_classification(model, num_classes=2):
    """
    Takes a pretrained GPT-2 backbone and modifies it for classification.
    Does NOT load weights. Does NOT build the model.
    """

    # 1. Freeze entire GPT‑2 backbone
    for p in model.parameters():
        p.requires_grad = False

    # 2. Replace LM head with classification head
    model.out_head = nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )

    # Make classifier head trainable
    for p in model.out_head.parameters():
        p.requires_grad = True

    # 3. Unfreeze last transformer block
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True

    # 4. Unfreeze final LayerNorm
    for p in model.final_norm.parameters():
        p.requires_grad = True

    print("\n=== GPT‑2 Classification Model Ready ===")
    print("Backbone frozen")
    print("Classification head attached\n")
    print("Last transformer block unfrozen")
    print("Final LayerNorm unfrozen")

    return model
