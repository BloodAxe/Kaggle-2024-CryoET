from dask.array import spacing
from monai.networks.nets import DynUNet
from torch import nn

from transformers import PretrainedConfig


class DynUNetForObjectDetectionConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        strides: int = 2,
        norm_name: str = "instance",
        deep_supervision: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.norm_name = norm_name
        self.deep_supervision = deep_supervision


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


class DynUNetForObjectDetection(nn.Module):
    def __init__(self, config: DynUNetForObjectDetectionConfig):
        super().__init__()
        k, s = get_kernels_strides([128, 128, 128], [1, 1, 1])
        kernels = [[3, 3, 3]] * 4
        strides = [[1, 1, 1]] + [[2, 2, 2]] * 3

        self.model = DynUNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            strides=s,
            upsample_kernel_size=s[:1],
            kernel_size=k,
            norm_name=config.norm_name,
            deep_supervision=config.deep_supervision,
        )

    def forward(self, x):
        return self.model(x)
