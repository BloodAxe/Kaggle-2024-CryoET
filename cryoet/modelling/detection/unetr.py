import torch
from monai.networks.nets import SwinUNETR
from torch import nn
from transformers import PretrainedConfig

from cryoet.modelling.detection.detection_head import ObjectDetectionOutput, ObjectDetectionHead
from cryoet.modelling.detection.functional import object_detection_loss


class SwinUNETRForObjectDetectionConfig(PretrainedConfig):
    def __init__(
        self,
        spatial_dims=3,
        img_size=96,
        in_channels=1,
        out_channels=14,
        feature_size=48,
        num_classes=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.num_classes = num_classes


class SwinUNETRFeatureExtractor(SwinUNETR):
    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0, enc0)
        return dec1, dec0
        # logits = self.out(out)
        # return logits


class SwinUNETRForObjectDetection(nn.Module):
    def __init__(self, config: SwinUNETRForObjectDetectionConfig):
        super().__init__()
        self.backbone = SwinUNETRFeatureExtractor(
            spatial_dims=config.spatial_dims,
            img_size=config.img_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=config.feature_size,
            use_checkpoint=True,
            dropout_path_rate=0.1,
        )

        self.head2 = ObjectDetectionHead(
            in_channels=48,
            num_classes=config.num_classes,
            stride=2,
            intermediate_channels=48,
            offset_intermediate_channels=16,
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        [fm4, fm2] = self.backbone(volume)

        # output4 = self.head4(fm4)
        output2 = self.head2(fm2)

        if torch.jit.is_tracing():
            # logits4, offsets4 = output4
            # return (logits4, logits2), (offsets4, offsets2)
            logits2, offsets2 = output2
            return (logits2,), (offsets2,)

        logits = [output2.logits]
        offsets = [output2.offsets]
        strides = [self.head2.stride]

        # logits = [output4.logits, output2.logits]
        # offsets = [output4.offsets, output2.offsets]
        # strides = [self.head4.stride, self.head2.stride]

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = object_detection_loss(logits, offsets, strides, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, strides=strides, loss=loss, loss_dict=loss_dict)
