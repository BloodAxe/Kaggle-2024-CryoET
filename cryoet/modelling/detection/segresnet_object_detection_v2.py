from typing import List

import torch.jit
from monai.networks.blocks import ResBlock
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers import get_norm_layer, get_act_layer, Dropout
from monai.utils import UpsampleMode
from torch import nn, Tensor

from .detection_head import ObjectDetectionHead, ObjectDetectionOutput

from transformers import PretrainedConfig

from .functional import object_detection_loss


class SegResNetBackbone(nn.Module):
    """
    SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2) if i > 0 else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act) for _ in range(blocks_up[i])]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: Tensor, down_x: List[Tensor]) -> Tensor:
        feature_maps = []

        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            feature_maps.append(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x, feature_maps

    def forward(self, x: Tensor) -> Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x


class SegResNetForObjectDetectionV2Config(PretrainedConfig):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
        head_dropout_prob=0,
        num_classes=5,
        use_stride4: bool = True,
        use_stride2: bool = True,
        use_offset_head: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.use_stride4 = use_stride4
        self.use_stride2 = use_stride2
        self.use_offset_head = use_offset_head
        self.head_dropout_prob = head_dropout_prob


class SegResNetForObjectDetectionV2(nn.Module):
    def __init__(self, config: SegResNetForObjectDetectionV2Config):
        super().__init__()
        self.config = config
        self.backbone = SegResNetBackbone(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=config.init_filters,
            blocks_down=config.blocks_down,
            blocks_up=config.blocks_up,
            dropout_prob=config.dropout_prob,
        )
        self.dropout = nn.Dropout3d(config.head_dropout_prob)
        if self.config.use_stride2:
            self.head2 = ObjectDetectionHead(
                in_channels=64,
                num_classes=config.num_classes,
                stride=2,
                intermediate_channels=48,
                offset_intermediate_channels=16,
                use_offset_head=config.use_offset_head,
            )

        if self.config.use_stride4:
            self.head4 = ObjectDetectionHead(
                in_channels=128,
                num_classes=config.num_classes,
                stride=4,
                intermediate_channels=64,
                offset_intermediate_channels=32,
                use_offset_head=config.use_offset_head,
            )

    def forward(self, volume, labels=None, apply_loss_on_each_stride: bool = False, is_tracing=False, **loss_kwargs):
        _, feature_maps = self.backbone(volume)
        fm4, fm2 = feature_maps[-3], feature_maps[-2]

        logits = []
        offsets = []
        strides = []

        if self.config.use_stride4:
            logits4, offsets4 = self.head4(self.dropout(fm4))
            logits.append(logits4)
            offsets.append(offsets4)
            strides.append(self.head4.stride)

        if self.config.use_stride2:
            logits2, offsets2 = self.head2(self.dropout(fm2))
            logits.append(logits2)
            offsets.append(offsets2)
            strides.append(self.head2.stride)

        if is_tracing or torch.jit.is_tracing() or torch.jit.is_scripting():
            return logits, offsets

        loss = None
        loss_dict = None
        if labels is not None:
            loss = 0
            loss_dict = {}

            if apply_loss_on_each_stride:
                for l, o, s in zip(logits, offsets, strides):
                    loss_per_stride, loss_dict_per_stride = object_detection_loss(l, o, s, labels, **loss_kwargs)
                    loss = loss + loss_per_stride
                    for k, v in loss_dict_per_stride.items():
                        if k in loss_dict:
                            loss_dict[k] = loss_dict[k] + v
                        else:
                            loss_dict[k] = v
            else:
                loss, loss_dict = object_detection_loss(logits, offsets, strides, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, strides=strides, loss=loss, loss_dict=loss_dict)
