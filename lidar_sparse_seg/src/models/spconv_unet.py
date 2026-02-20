from __future__ import annotations

import torch.nn as nn
import spconv.pytorch as spconv


def _replace_feature(x: spconv.SparseConvTensor, features):
    return x.replace_feature(features)


class SubMConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, indice_key: str) -> None:
        super().__init__()
        self.conv = spconv.SubMConv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv(x)
        out = _replace_feature(out, self.bn(out.features))
        out = _replace_feature(out, self.act(out.features))
        return out


class SparseResidualBlock(nn.Module):
    def __init__(self, channels: int, indice_key: str) -> None:
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x.features

        out = self.conv1(x)
        out = _replace_feature(out, self.bn1(out.features))
        out = _replace_feature(out, self.act(out.features))

        out = self.conv2(out)
        out = _replace_feature(out, self.bn2(out.features))
        out = _replace_feature(out, out.features + identity)
        out = _replace_feature(out, self.act(out.features))
        return out


class SparseUNet(nn.Module):
    """Lightweight sparse segmentation net on original voxel resolution (spconv backend)."""

    def __init__(self, in_channels: int = 2, num_classes: int = 5, base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.stem = SubMConvBNReLU(in_channels, base_channels, indice_key="subm_stem")
        self.blocks = nn.ModuleList(
            [SparseResidualBlock(base_channels, indice_key=f"subm_block_{i}") for i in range(depth)]
        )
        self.head = spconv.SubMConv3d(
            base_channels,
            num_classes,
            kernel_size=1,
            padding=0,
            bias=True,
            indice_key="subm_head",
        )

    def forward(self, x: spconv.SparseConvTensor):
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        logits = self.head(out)
        return logits.features
