from __future__ import annotations

import torch
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


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, indice_key: str) -> None:
        super().__init__()
        self.down = spconv.SparseConv3d(
            in_ch,
            out_ch,
            kernel_size=(2, 2, 1),
            stride=(2, 2, 1),
            padding=0,
            bias=False,
            indice_key=indice_key,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.refine = SparseResidualBlock(out_ch, indice_key=f"{indice_key}_subm")

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.down(x)
        out = _replace_feature(out, self.bn(out.features))
        out = _replace_feature(out, self.act(out.features))
        out = self.refine(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, block_key: str) -> None:
        super().__init__()
        # Transposed sparse conv is more robust than SparseInverseConv3d here
        # because it does not require pre-existing indice-pair cache.
        self.up = spconv.SparseConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=(2, 2, 1),
            stride=(2, 2, 1),
            padding=0,
            bias=False,
        )
        self.up_bn = nn.BatchNorm1d(out_ch)
        self.up_act = nn.ReLU(inplace=True)
        self.fuse = SubMConvBNReLU(out_ch + skip_ch, out_ch, indice_key=f"{block_key}_fuse")
        self.refine = SparseResidualBlock(out_ch, indice_key=f"{block_key}_refine")

    def _merge_skip(
        self,
        up_x: spconv.SparseConvTensor,
        skip_x: spconv.SparseConvTensor,
    ) -> spconv.SparseConvTensor:
        same_layout = (
            up_x.indices.shape == skip_x.indices.shape
            and torch.equal(up_x.indices, skip_x.indices)
            and up_x.spatial_shape == skip_x.spatial_shape
            and up_x.batch_size == skip_x.batch_size
        )
        if same_layout:
            merged = torch.cat([up_x.features, skip_x.features], dim=1)
            return spconv.SparseConvTensor(
                features=merged,
                indices=up_x.indices,
                spatial_shape=up_x.spatial_shape,
                batch_size=up_x.batch_size,
            )

        # Fallback when sparse layouts differ: keep decoder branch only.
        pad = torch.zeros(
            (up_x.features.shape[0], skip_x.features.shape[1]),
            dtype=up_x.features.dtype,
            device=up_x.features.device,
        )
        padded = torch.cat(
            [up_x.features, pad],
            dim=1,
        )
        return spconv.SparseConvTensor(
            features=padded,
            indices=up_x.indices,
            spatial_shape=up_x.spatial_shape,
            batch_size=up_x.batch_size,
        )

    def forward(
        self,
        x: spconv.SparseConvTensor,
        skip: spconv.SparseConvTensor,
    ) -> spconv.SparseConvTensor:
        out = self.up(x)
        out = _replace_feature(out, self.up_bn(out.features))
        out = _replace_feature(out, self.up_act(out.features))
        out = self._merge_skip(out, skip)
        out = self.fuse(out)
        out = self.refine(out)
        return out


class SparseUNetMultiScale(nn.Module):
    """Encoder-decoder sparse UNet with downsample/upsample and skip connections."""

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 5,
        base_channels: int = 32,
        depth: int = 2,
    ) -> None:
        super().__init__()
        blocks = max(int(depth), 1)

        self.stem = SubMConvBNReLU(in_channels, base_channels, indice_key="ms_stem")
        self.enc_refine = nn.Sequential(
            *[SparseResidualBlock(base_channels, indice_key=f"ms_enc0_{i}") for i in range(blocks)]
        )

        self.down1 = DownBlock(base_channels, base_channels * 2, indice_key="ms_down1")
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, indice_key="ms_down2")

        self.bottleneck = nn.Sequential(
            *[SparseResidualBlock(base_channels * 4, indice_key=f"ms_bottleneck_{i}") for i in range(blocks)]
        )

        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, block_key="ms_up2")
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels, block_key="ms_up1")
        self.classifier = nn.Linear(base_channels, num_classes)

    @staticmethod
    def _coords_hash(indices: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
        idx = indices.to(torch.int64)
        return (
            idx[:, 0]
            + idx[:, 1] * bases[0]
            + idx[:, 2] * bases[1]
            + idx[:, 3] * bases[2]
        )

    @staticmethod
    def _align_features_to_ref(
        src: spconv.SparseConvTensor,
        ref: spconv.SparseConvTensor,
    ) -> torch.Tensor:
        """Project src sparse features to reference coordinates (missing -> zeros)."""
        src_idx = src.indices
        ref_idx = ref.indices
        if src_idx.shape[0] == 0:
            return torch.zeros(
                (ref_idx.shape[0], src.features.shape[1]),
                dtype=src.features.dtype,
                device=src.features.device,
            )

        max_vals = torch.maximum(src_idx.max(dim=0).values, ref_idx.max(dim=0).values).to(torch.int64) + 1
        bases = torch.tensor(
            [max_vals[0], max_vals[0] * max_vals[1], max_vals[0] * max_vals[1] * max_vals[2]],
            dtype=torch.int64,
            device=src.features.device,
        )

        src_hash = SparseUNetMultiScale._coords_hash(src_idx, bases)
        ref_hash = SparseUNetMultiScale._coords_hash(ref_idx, bases)

        src_hash_sorted, src_order = torch.sort(src_hash)
        pos = torch.searchsorted(src_hash_sorted, ref_hash)
        in_range = pos < src_hash_sorted.shape[0]
        valid = torch.zeros_like(in_range, dtype=torch.bool)
        if torch.any(in_range):
            in_range_pos = pos[in_range]
            valid[in_range] = src_hash_sorted[in_range_pos] == ref_hash[in_range]

        out = torch.zeros(
            (ref_idx.shape[0], src.features.shape[1]),
            dtype=src.features.dtype,
            device=src.features.device,
        )
        if torch.any(valid):
            out[valid] = src.features[src_order[pos[valid]]]
        return out

    def forward(self, x: spconv.SparseConvTensor):
        x0 = self.stem(x)
        x0 = self.enc_refine(x0)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x2 = self.bottleneck(x2)

        y1 = self.up2(x2, x1)
        y0 = self.up1(y1, x0)
        y0_aligned = self._align_features_to_ref(y0, x0)
        return self.classifier(y0_aligned)
