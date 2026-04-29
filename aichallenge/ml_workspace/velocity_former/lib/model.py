"""VelocityFormer PyTorch model.

Wraps a HuggingFace BERT model (default: prajjwal1/bert-tiny) and adds a
regression head that predicts a single scalar (velocity or steering) from
trajectory-derived integer "tokens".
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class VelocityFormer(nn.Module):
    """BERT-based regression model that predicts a control value from a trajectory.

    Inputs are integer ID sequences of length `input_size`, derived from the
    angle (degrees) between adjacent trajectory points. The output is a single
    scalar (velocity [m/s] or steering [rad]).
    """

    def __init__(
        self,
        pretrained_model: str = "prajjwal1/bert-tiny",
        input_size: int = 12,
        num_labels: int = 1,
        load_pretrained_weights: bool = True,
    ):
        """Initialize VelocityFormer.

        Args:
            pretrained_model: HuggingFace model id used both for config and (optionally) weights.
            input_size: Sequence length (number of trajectory tokens).
            num_labels: Output dimension (1 for scalar regression).
            load_pretrained_weights: If True, load pretrained encoder weights from HF Hub.
        """
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(pretrained_model)
        if load_pretrained_weights:
            self.bert = AutoModel.from_pretrained(pretrained_model)
        else:
            self.bert = AutoModel.from_config(config)

        self.head = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the forward pass.

        Args:
            input_ids: Long tensor of shape (B, input_size) in [0, vocab_size).
            attention_mask: Optional bool/int mask of shape (B, input_size).

        Returns:
            Tensor of shape (B, num_labels) with the predicted control value.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # max-pool over sequence dimension (matches the original VelocityFormer)
        pooled, _ = last_hidden_state.max(dim=1)
        return self.head(pooled)
