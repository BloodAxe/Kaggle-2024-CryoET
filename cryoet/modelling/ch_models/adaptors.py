import einops
import torch
from torch import nn, Tensor

from .mdl_ch_20d_ce2 import FlexibleUNet
from .mdl_ch_20d_ce2c2 import FlexibleUNet as FlexibleUNet2c2


class FakeObjectDetectionAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def mean_std_renormalization(self, volume: Tensor):
        """
        Renormalize the volume to have zero mean and unit variance.
        :param volume: Tensor of shape (B, C, D, H, W)
        """
        volume = volume
        mean = volume.mean(dim=(1, 2, 3, 4), keepdim=True)
        std = volume.std(dim=(1, 2, 3, 4), keepdim=True)
        volume = (volume - mean) / std
        return volume

    def forward(self, volume, is_tracing=False):
        volume = self.mean_std_renormalization(volume)  # Adapt to different normalization
        volume = einops.rearrange(volume, "B C D H W -> B C W H D")  # .transpose(2,1,0) but more clear what is happening

        logits = self.forward_backbone(volume)

        probas = logits.softmax(1)  # 7 classes [Background, then 6 classes] ?
        probas = probas[:, [0, 2, 3, 4, 5, 1]]  # Reorder classes

        probas = einops.rearrange(probas, "B C W H D -> B C D H W")

        # probas = torch.nn.functional.interpolate(probas, scale_factor=2, mode="trilinear")
        # probas = torch.nn.functional.interpolate(probas, scale_factor=0.5, mode="trilinear")

        fake_offsets = torch.zeros_like(probas[:, 0:3])
        if is_tracing:
            return [probas], [fake_offsets]
        return probas, fake_offsets

    def forward_backbone(self, volume):
        out = self.backbone(volume)
        logits = out[-1]
        return logits


class MdlCh20dCe2_resnet34(FakeObjectDetectionAdapter):
    def __init__(self):
        super().__init__()
        backbone_args = dict(spatial_dims=3, in_channels=1, out_channels=6, backbone="resnet34", pretrained=False)
        self.backbone = FlexibleUNet(**backbone_args)


class MdlCh20dCe2c2_resnet34(FakeObjectDetectionAdapter):
    def __init__(self):
        super().__init__()
        backbone_args = dict(spatial_dims=3, in_channels=1, out_channels=6, backbone="resnet34", pretrained=False)
        self.backbone = FlexibleUNet2c2(**backbone_args)


class MdlCh20dCe2_effnetb3(FakeObjectDetectionAdapter):
    def __init__(self):
        super().__init__()
        backbone_args = dict(spatial_dims=3, in_channels=1, out_channels=6, backbone="efficientnet-b3", pretrained=False)
        self.backbone = FlexibleUNet(**backbone_args)
