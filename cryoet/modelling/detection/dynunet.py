import torch
from monai.networks.nets import DynUNet
from torch import nn

from transformers import PretrainedConfig

from cryoet.modelling.detection.detection_head import ObjectDetectionHead, ObjectDetectionOutput
from cryoet.modelling.detection.functional import object_detection_loss


def get_kernels_strides(sizes, spacings):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


class DynUNetFeatureExtractor(DynUNet):

    def forward(self, x):
        _ = self.skip_layers(x)
        # h = self.heads
        # out = self.output_block(out)
        return self.heads


class DynUNetForObjectDetectionConfig(PretrainedConfig):
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        out_channels: int = 64,
        norm_name: str = "instance",
        use_stride4: bool = True,
        use_stride2: bool = True,
        res_block: bool = False,
        act_name: str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout=None,
        intermediate_channels=48,
        offset_intermediate_channels=16,
        object_size: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.object_size = object_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_name = norm_name
        self.use_stride4 = use_stride4
        self.use_stride2 = use_stride2
        self.res_block = res_block
        self.act_name = act_name
        self.dropout = dropout
        self.intermediate_channels = intermediate_channels
        self.offset_intermediate_channels = offset_intermediate_channels


class DynUNetForObjectDetection(nn.Module):
    def __init__(self, config: DynUNetForObjectDetectionConfig):
        super().__init__()
        self.config = config
        kernels, strides = get_kernels_strides([config.object_size, config.object_size, config.object_size], [1, 1, 1])

        # kernels = [[3, 3, 3]] * 4
        # strides = [[1, 1, 1]] + [[2, 2, 2]] * 3

        self.backbone = DynUNetFeatureExtractor(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name=config.norm_name,
            deep_supervision=True,
            deep_supr_num=2,
            dropout=config.dropout,
            res_block=config.res_block,
            act_name=config.act_name,
        )

        if self.config.use_stride2:
            self.head2 = ObjectDetectionHead(
                in_channels=config.out_channels,
                num_classes=config.num_classes,
                stride=2,
                intermediate_channels=config.intermediate_channels,
                offset_intermediate_channels=config.offset_intermediate_channels,
            )

        if self.config.use_stride4:
            self.head4 = ObjectDetectionHead(
                in_channels=config.out_channels,
                num_classes=config.num_classes,
                stride=4,
                intermediate_channels=config.intermediate_channels,
                offset_intermediate_channels=config.offset_intermediate_channels,
            )

    def forward(self, volume, labels=None, **loss_kwargs):
        [fm2, fm4] = self.backbone(volume)

        logits = []
        offsets = []
        strides = []

        if self.config.use_stride4:
            logits4, offsets4 = self.head4(fm4)
            logits.append(logits4)
            offsets.append(offsets4)
            strides.append(self.head4.stride)

        if self.config.use_stride2:
            logits2, offsets2 = self.head2(fm2)
            logits.append(logits2)
            offsets.append(offsets2)
            strides.append(self.head2.stride)

        if torch.jit.is_tracing():
            return logits, offsets

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = object_detection_loss(logits, offsets, strides, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, strides=strides, loss=loss, loss_dict=loss_dict)
