"""双层 LSTM。每一帧都有类别，因果，不看未来。"""
from __future__ import annotations

import torch
from torch import nn


class ActionLSTM(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch, lengths=None, all_steps: bool = False):
        output, _hidden = self.lstm(batch)
        logits = self.fc(output)
        if all_steps:
            return logits
        if lengths is None:
            return logits[:, -1, :]
        # 右端补齐的位置不能当成这段的结尾。取每一条真实的最后一帧。
        index = torch.as_tensor(lengths, device=logits.device) - 1
        return logits[torch.arange(logits.shape[0], device=logits.device), index]

    def step(self, batch, hidden=None):
        """逐段前向并交还隐藏状态。不传 hidden 时与从零开始的 forward 相同。"""
        output, hidden_out = self.lstm(batch, hidden)
        return self.fc(output), hidden_out

    def stage_logits(self, batch, lengths=None):
        return [self.forward(batch, lengths, all_steps=True)]
