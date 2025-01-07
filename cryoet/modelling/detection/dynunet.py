import torch
from monai.networks.nets import DynUNet
from torch import nn

from transformers import PretrainedConfig

from cryoet.modelling.detection.detection_head import ObjectDetectionHead, ObjectDetectionOutput
from cryoet.modelling.detection.functional import object_detection_loss


class DynUNetForObjectDetectionConfig(PretrainedConfig):
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        out_channels: int = 64,
        norm_name: str = "instance",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_name = norm_name


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


class DynUNetForObjectDetection(nn.Module):
    def __init__(self, config: DynUNetForObjectDetectionConfig):
        super().__init__()
        kernels, strides = get_kernels_strides([128, 128, 128], [1, 1, 1])

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
            deep_supr_num=1,
        )

        self.head2 = ObjectDetectionHead(
            in_channels=config.out_channels,
            num_classes=config.num_classes,
            stride=2,
            intermediate_channels=48,
            offset_intermediate_channels=16,
        )
        # self.head4 = ObjectDetectionHead(
        #     in_channels=config.out_channels,
        #     num_classes=config.num_classes,
        #     stride=4,
        #     intermediate_channels=48,
        #     offset_intermediate_channels=16,
        # )

    def forward(self, volume, labels=None, **loss_kwargs):
        [fm2] = self.backbone(volume)

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
