from __future__ import annotations

from src.models.spconv_unet import SparseUNet
from src.models.spconv_unet_multiscale import SparseUNetMultiScale


def build_model(cfg: dict, in_channels: int, num_classes: int):
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "spconv_unet")).lower().strip()
    base_channels = int(model_cfg.get("base_channels", 32))
    depth = int(model_cfg.get("depth", 4))

    if model_name == "spconv_unet":
        return SparseUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
        )
    if model_name == "spconv_unet_multiscale":
        return SparseUNetMultiScale(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
        )
    raise ValueError(f"Unsupported model.name='{model_name}'. Use spconv_unet or spconv_unet_multiscale.")
