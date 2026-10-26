"""Embedding head for the identity embeddings.

3-layer MLP that maps M2F decoder query embeddings to a contrastive
embedding space. Architecture follows CTVIS (NUM_REID_HEAD_LAYERS=3,
REID_HIDDEN_DIM=256) and the standard MLP used in Mask2Former/DVIS++
(Linear → ReLU → Linear → ReLU → Linear, no final activation).

Input:  (*, C=256) decoder query embeddings
Output: (*, C=256) contrastive embeddings for InfoNCE loss
"""

import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """3-layer MLP for contrastive embeddings.

    Args:
        input_dim: Input dimension (M2F decoder hidden_dim). Default: 256.
        hidden_dim: Hidden layer dimension. Default: 256.
        output_dim: Output embedding dimension. Default: 256.
        num_layers: Number of linear layers. Default: 3.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization (matches Mask2Former MLP convention)."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (*, C) tensor of decoder query embeddings.

        Returns:
            (*, C) tensor of contrastive embeddings.
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
