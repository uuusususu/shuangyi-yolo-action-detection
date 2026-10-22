"""双层 LSTM。每一帧都有类别，因果，不看未来。"""
from __future__ import annotations

import torch
from torch import nn


def receptive_field(layers: int, stages: int = 2) -> int:
    """因果空洞卷积往回看的帧数。一层贡献 2×空洞率，两阶段首尾相接。"""
    one = 1 + 2 * ((1 << int(layers)) - 1)
    if stages <= 1:
        return one
    return 1 + int(stages) * (one - 1)


def mstcn_layers_for(context_frames: int) -> int:
    """4 到 10 层里，两阶段往回看不超过上文长度的最大层数。最短也用 4 层。"""
    chosen = 4
    for layers in range(4, 11):
        if receptive_field(layers, 2) <= int(context_frames):
            chosen = layers
        else:
            break
    return chosen


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


class CausalMSTCN(nn.Module):
    """精简因果 MS-TCN。只看已经发生的帧，后一阶段修前一阶段的边界。"""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        channels: int = 64,
        layers: int = 8,
        stages: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stages = nn.ModuleList([
            _MSTCNStage(input_size if index == 0 else num_classes, channels, num_classes, layers, dropout)
            for index in range(stages)
        ])

    def stage_logits(self, batch, lengths=None):
        del lengths
        x = batch.transpose(1, 2)
        outputs = []
        for index, stage in enumerate(self.stages):
            logits = stage(x)
            outputs.append(logits.transpose(1, 2))
            if index + 1 < len(self.stages):
                x = torch.softmax(logits, dim=1)
        return outputs

    def forward(self, batch, lengths=None, all_steps: bool = False):
        logits = self.stage_logits(batch)[-1]
        if all_steps:
            return logits
        if lengths is None:
            return logits[:, -1, :]
        index = torch.as_tensor(lengths, device=logits.device) - 1
        return logits[torch.arange(logits.shape[0], device=logits.device), index]


class _DualDilated(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.narrow = max(1, dilation // 2)
        self.conv_wide = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv_narrow = nn.Conv1d(channels, channels, 3, dilation=self.narrow)
        self.drop = nn.Dropout(dropout)

    def _causal(self, x, conv, dilation: int):
        return conv(nn.functional.pad(x, (2 * dilation, 0)))

    def forward(self, x):
        mixed = self._causal(x, self.conv_wide, self.dilation) + self._causal(x, self.conv_narrow, self.narrow)
        return nn.functional.relu(x + self.drop(mixed))


class _MSTCNStage(nn.Module):
    def __init__(self, in_channels: int, channels: int, num_classes: int, layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, channels, 1)
        self.layers = nn.ModuleList([_DualDilated(channels, 2 ** index, dropout) for index in range(layers)])
        self.pred = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x):
        hidden = self.proj(x)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.pred(hidden)
