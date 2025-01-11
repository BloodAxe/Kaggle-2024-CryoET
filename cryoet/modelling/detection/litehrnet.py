import torch
import torch.nn as nn
from pytorch_toolbelt.utils import count_parameters
from pytorch_toolbelt.modules import get_activation_block
from torch.nn import Conv3d, BatchNorm3d, Upsample, ConvTranspose2d
from transformers import PretrainedConfig

from cryoet.modelling.detection.detection_head import ObjectDetectionHead, ObjectDetectionOutput
from cryoet.modelling.detection.functional import object_detection_loss


def initialize_weights(m):
    if isinstance(m, (nn.Conv3d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


class HRNetConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 16,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.activation = activation


class ResidualBasic(torch.nn.Module):
    """
    Residual block
    input_width (int): Number of input conv channels,
    retained through sequentially conducted blocks in one branch.
    """

    def __init__(self, in_channels: int, activation_block):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.conv1 = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm3d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm3d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = activation_block()

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.act(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o += x
        o = self.act(o)
        return o


class FirstResidualBottleneck(torch.nn.Module):
    """
    1'st Residual block with bottleneck
    with hardcoded out channels,
    input_width (int): Number of input conv channels,
    which are squeezed into 64 for processing by 3x3 conv3
    and then unsqueezed back for output.
    """

    def __init__(self, in_channels, bottleneck_channels: int, out_channels: int, activation_block):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.conv1 = Conv3d(self.in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm3d(bottleneck_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm3d(bottleneck_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv3d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = activation_block()
        self.fcconv1 = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.fcconv1bn1 = BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.act(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.act(o)
        o = self.conv3(o)
        o = self.bn3(o)
        o += self.fcconv1bn1(self.fcconv1(x))
        o = self.act(o)
        return o


class ResidualBottleneck(torch.nn.Module):
    """
    Residual block with bottleneck
    input_width (int): Number of input conv channels,
    which are squeezed into 64 for processing by 3x3 conv3
    and then unsqueezed back for output.
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, activation_block):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = in_channels

        self.conv1 = Conv3d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm3d(bottleneck_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2 = Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm3d(bottleneck_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3 = Conv3d(bottleneck_channels, self.in_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm3d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.act = activation_block()

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.act(o)

        o = self.conv2(o)
        o = self.bn2(o)
        o = self.act(o)

        o = self.conv3(o)
        o = self.bn3(o)
        o += x
        o = self.act(o)
        return o


class Transition(torch.nn.Module):
    """
    Transition block
    input_width (int): Number of input conv channels (from upper branch).
    output_width (int): Number of output conv channels (to lower branch).
    downsample (bool): True when feature map is transited to lower branch. Default: None.
    """

    def __init__(self, input_width, output_width, activation_block, downsample=None):
        super().__init__()
        self.i_N = input_width
        self.o_N = output_width
        self.downsample = downsample
        if self.downsample:
            self.conv1 = Conv3d(self.i_N, self.o_N, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = Conv3d(self.i_N, self.o_N, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm3d(self.o_N, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = activation_block()

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.act(o)
        return o


class Fusion(torch.nn.Module):
    """
    Fusion block
    input_width (int): Number of input conv channels (from feature map used for fusion)
    output_width (int): Number of output conv channels (from target feature map to fuse with)
    upsample (bool): True when feature map is fused with feature map from upper branch. Default: None.
    downsample (bool): True when feature map is fused with feature map from lower branch. Default: None.
    """

    def __init__(self, input_width, output_width, activation_block, scale_factor=2, upsample=None, downsample=None):
        super().__init__()
        self.i_N = input_width
        self.o_N = output_width
        self.upsample = upsample
        self.downsample = downsample
        self.scale_factor = scale_factor
        if self.upsample:
            assert downsample == None
            self.conv1 = Conv3d(self.i_N, self.o_N, kernel_size=3, stride=1, padding=1, bias=False)
            self.upsample1 = Upsample(scale_factor=self.scale_factor)
        if self.downsample:
            assert upsample == None
            self.conv1 = Conv3d(self.i_N, self.o_N, kernel_size=3, stride=2, padding=1, bias=False)
            self.act = activation_block()
        self.bn1 = BatchNorm3d(self.o_N, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        if self.upsample:
            o = self.conv1(x)
            o = self.bn1(o)
            o = self.upsample1(o)
        if self.downsample:
            o = self.conv1(x)
            o = self.bn1(o)
            o = self.act(o)
        return o


class HRModule(torch.nn.Module):
    """Multi-Resolution Module for HRNet.
    In this module, every branch has 4 BasicBlocks/Bottlenecks.
    stage (int): Num of stage of HRNetv2.
    transition_stage (bool): True when HRModule of stage 3 or 4 should use Transition block,
    (stage 1 and 2 use it by default and controlled by this param). Default: None.
    """

    def __init__(self, config: HRNetConfig, stage: int, transition_stage=None):
        super().__init__()
        activation_block = get_activation_block(config.activation)
        self.stage = stage
        self.transition_stage = transition_stage
        if self.stage == 1:
            self.s1_b1_block1 = FirstResidualBottleneck(
                config.stem_channels, bottleneck_channels=48, out_channels=32, activation_block=activation_block
            )
            self.s1_b1_block2 = ResidualBottleneck(32, bottleneck_channels=48, activation_block=activation_block)
            self.s1_b1_block3 = ResidualBottleneck(32, bottleneck_channels=48, activation_block=activation_block)
            self.s1_b1_block4 = ResidualBottleneck(32, bottleneck_channels=48, activation_block=activation_block)
            # to 18
            self.transition_s1_256_18 = Transition(self.s1_b1_block4.out_channels, 18, activation_block=activation_block)
            # to 36
            self.transition_s1_256_36 = Transition(
                self.s1_b1_block4.out_channels, 36, downsample=True, activation_block=activation_block
            )
        if self.stage == 2:
            self.s2_b1_block1 = ResidualBasic(18, activation_block=activation_block)
            self.s2_b1_block2 = ResidualBasic(18, activation_block=activation_block)
            self.s2_b1_block3 = ResidualBasic(18, activation_block=activation_block)
            self.s2_b1_block4 = ResidualBasic(18, activation_block=activation_block)
            # to 36
            self.fuse_s2_18_36 = Fusion(18, 36, downsample=True, activation_block=activation_block)
            # to 72
            self.transition_s2_18 = Transition(18, 18, downsample=True, activation_block=activation_block)
            self.transition_s2_18_72 = Transition(18, 72, downsample=True, activation_block=activation_block)

            self.s2_b2_block1 = ResidualBasic(36, activation_block=activation_block)
            self.s2_b2_block2 = ResidualBasic(36, activation_block=activation_block)
            self.s2_b2_block3 = ResidualBasic(36, activation_block=activation_block)
            self.s2_b2_block4 = ResidualBasic(36, activation_block=activation_block)
            # to 18
            self.fuse_s2_36_18 = Fusion(36, 18, upsample=True, activation_block=activation_block)
            # to 72
            self.transition_s2_36_72 = Transition(36, 72, downsample=True, activation_block=activation_block)
        if self.stage == 3:
            self.s3_b1_block1 = ResidualBasic(18, activation_block=activation_block)
            self.s3_b1_block2 = ResidualBasic(18, activation_block=activation_block)
            self.s3_b1_block3 = ResidualBasic(18, activation_block=activation_block)
            self.s3_b1_block4 = ResidualBasic(18, activation_block=activation_block)
            # to 36
            self.down_fuse_s3_18_36 = Fusion(18, 36, downsample=True, activation_block=activation_block)
            # to 72
            self.down_fuse_s3_18 = Fusion(18, 18, downsample=True, activation_block=activation_block)
            self.down_fuse_s3_18_72 = Fusion(18, 72, downsample=True, activation_block=activation_block)

            self.s3_b2_block1 = ResidualBasic(36, activation_block=activation_block)
            self.s3_b2_block2 = ResidualBasic(36, activation_block=activation_block)
            self.s3_b2_block3 = ResidualBasic(36, activation_block=activation_block)
            self.s3_b2_block4 = ResidualBasic(36, activation_block=activation_block)
            # to 18
            self.up_fuse_s3_36_18 = Fusion(36, 18, upsample=True, activation_block=activation_block)
            # to 72
            self.down_fuse_s3_36_72 = Fusion(36, 72, downsample=True, activation_block=activation_block)

            self.s3_b3_block1 = ResidualBasic(72, activation_block=activation_block)
            self.s3_b3_block2 = ResidualBasic(72, activation_block=activation_block)
            self.s3_b3_block3 = ResidualBasic(72, activation_block=activation_block)
            self.s3_b3_block4 = ResidualBasic(72, activation_block=activation_block)
            # to 18
            self.up_fuse_s3_72_72 = Fusion(72, 72, upsample=True, activation_block=activation_block)
            self.up_fuse_s3_72_18 = Fusion(72, 18, upsample=True, activation_block=activation_block)
            # to 36
            self.up_fuse_s3_72_36 = Fusion(72, 36, upsample=True, activation_block=activation_block)

            if self.transition_stage:
                # 18 to 144
                self.transition_s3_18_36 = Transition(18, 36, downsample=True, activation_block=activation_block)
                self.transition1_s3_36_72 = Transition(36, 72, downsample=True, activation_block=activation_block)
                self.transition1_s3_72_144 = Transition(72, 144, downsample=True, activation_block=activation_block)
                # 36 to 144
                self.transition2_s3_36_72 = Transition(36, 72, downsample=True, activation_block=activation_block)
                self.transition2_s3_72_144 = Transition(72, 144, downsample=True, activation_block=activation_block)
                # 72 to 144
                self.transition3_s3_72_144 = Transition(72, 144, downsample=True, activation_block=activation_block)
        if self.stage == 4:
            self.s4_b1_block1 = ResidualBasic(18, activation_block=activation_block)
            self.s4_b1_block2 = ResidualBasic(18, activation_block=activation_block)
            self.s4_b1_block3 = ResidualBasic(18, activation_block=activation_block)
            self.s4_b1_block4 = ResidualBasic(18, activation_block=activation_block)
            # to 36
            self.down_fuse_s4_18_36 = Fusion(18, 36, downsample=True, activation_block=activation_block)
            # to 72
            self.down1_fuse_s4_18 = Fusion(18, 18, downsample=True, activation_block=activation_block)
            self.down_fuse_s4_18_72 = Fusion(18, 72, downsample=True, activation_block=activation_block)
            # to 144
            self.down2_fuse_s4_18 = Fusion(18, 18, downsample=True, activation_block=activation_block)
            self.down1_fuse_s4_18_144 = Fusion(18, 18, downsample=True, activation_block=activation_block)
            self.down2_fuse_s4_18_144 = Fusion(18, 144, downsample=True, activation_block=activation_block)

            self.s4_b2_block1 = ResidualBasic(36, activation_block=activation_block)
            self.s4_b2_block2 = ResidualBasic(36, activation_block=activation_block)
            self.s4_b2_block3 = ResidualBasic(36, activation_block=activation_block)
            self.s4_b2_block4 = ResidualBasic(36, activation_block=activation_block)
            # to 18
            self.up_fuse_s4_36_18 = Fusion(36, 18, upsample=True, activation_block=activation_block)
            # to 72
            self.down_fuse_s4_36_72 = Fusion(36, 72, downsample=True, activation_block=activation_block)
            # to 144
            self.down2_fuse_s4_36 = Fusion(36, 36, downsample=True, activation_block=activation_block)
            self.down_fuse_s4_36_144 = Fusion(36, 144, downsample=True, activation_block=activation_block)

            self.s4_b3_block1 = ResidualBasic(72, activation_block=activation_block)
            self.s4_b3_block2 = ResidualBasic(72, activation_block=activation_block)
            self.s4_b3_block3 = ResidualBasic(72, activation_block=activation_block)
            self.s4_b3_block4 = ResidualBasic(72, activation_block=activation_block)
            # to 18
            self.up_fuse_s4_72_72 = Fusion(72, 72, upsample=True, activation_block=activation_block)
            self.up_fuse_s4_72_18 = Fusion(72, 18, upsample=True, activation_block=activation_block)
            # to 36
            self.up_fuse_s4_72_36 = Fusion(72, 36, upsample=True, activation_block=activation_block)
            # to 144
            self.up_fuse_s4_72_144 = Fusion(72, 144, downsample=True, activation_block=activation_block)

            self.s4_b4_block1 = ResidualBasic(144, activation_block=activation_block)
            self.s4_b4_block2 = ResidualBasic(144, activation_block=activation_block)
            self.s4_b4_block3 = ResidualBasic(144, activation_block=activation_block)
            self.s4_b4_block4 = ResidualBasic(144, activation_block=activation_block)
            # to 18
            self.up1_fuse_s4_144_144 = Fusion(144, 144, upsample=True, activation_block=activation_block)
            self.up2_fuse_s4_144_144 = Fusion(144, 144, upsample=True, activation_block=activation_block)
            self.up_fuse_s4_144_18 = Fusion(144, 18, upsample=True, activation_block=activation_block)
            # to 36
            self.up3_fuse_s4_144_144 = Fusion(144, 144, upsample=True, activation_block=activation_block)
            self.up_fuse_s4_144_36 = Fusion(144, 36, upsample=True, activation_block=activation_block)
            # to 72
            self.up_fuse_s4_144_72 = Fusion(144, 72, upsample=True, activation_block=activation_block)

        self.relu = activation_block()

    def forward(self, x):
        """
        x (list): list of inputs (len=1 at stage=1, len>1 at stages>1)
        """
        outputs = []
        if self.stage == 1:
            # at this stage, 1 HRModule should be initialized
            # x[0] = 64 channels feature map from backbone beginning
            o = self.s1_b1_block1(x[0])
            o = self.s1_b1_block2(o)
            o = self.s1_b1_block3(o)
            o = self.s1_b1_block4(o)
            # outputs of stage 1: [18 channels feature map,
            #                       36 channels feature map]
            outputs.append(self.transition_s1_256_18(o))
            outputs.append(self.transition_s1_256_36(o))
        if self.stage == 2:
            # at this stage, 1 HRModule should be initialized
            # x[0] = 18 channels feature map from stage 1 output
            o = self.s2_b1_block1(x[0])
            o = self.s2_b1_block2(o)
            o = self.s2_b1_block3(o)
            o_18 = self.s2_b1_block4(o)

            # x[1] = 36 channels feature map from stage 1 output
            o = self.s2_b2_block1(x[1])
            o = self.s2_b2_block2(o)
            o = self.s2_b2_block3(o)
            o_36 = self.s2_b2_block4(o)

            # to 18
            outputs.append(self.fuse_s2_36_18(o) + o_18)  #
            # to 36
            outputs.append(self.fuse_s2_18_36(o_18) + o_36)
            # fused (transited 18 to 72 + transited 36 to 72)
            outputs.append(self.transition_s2_18_72(self.transition_s2_18(o_18)) + self.transition_s2_36_72(o_36))

            # outputs of stage 2: [fused 18 channels feature map,
            #                       fused 36 channels feature map,
            #                       fused(transited 18 to 72 channels feature map
            #                       + transited 36 to 72 channels feature map)]
        if self.stage == 3:
            # at this stage, 4 subsequent HRModules should be initialized in model definition
            # x[0] = fused 18 channels feature map
            o = self.s3_b1_block1(x[0])
            o = self.s3_b1_block2(o)
            o = self.s3_b1_block3(o)
            o_18 = self.s3_b1_block4(o)

            # x[1] = fused 36 channels feature map
            o = self.s3_b2_block1(x[1])
            o = self.s3_b2_block2(o)
            o = self.s3_b2_block3(o)
            o_36 = self.s3_b2_block4(o)

            # x[2] = fused(transited 18 to 72 channels feature map + transited 36 to 72 channels feature map
            # (when first HRModule of 3 stage),
            # x[2] = fused 72 channels feature map (when 2,3 or 4 subsequent HRModule of 3 stage)
            o = self.s3_b3_block1(x[2])
            o = self.s3_b3_block2(o)
            o = self.s3_b3_block3(o)
            o_72 = self.s3_b3_block4(o)

            # to 18 (fused 36 and 72 to 18)
            outputs.append(self.up_fuse_s3_36_18(o_36) + self.up_fuse_s3_72_18(self.up_fuse_s3_72_72(o_72)))
            # to 36 (fused 18 and 72 to 36)
            outputs.append(self.down_fuse_s3_18_36(o_18) + self.up_fuse_s3_72_36(o_72))
            # to 72 (fused 18 and 36 to 72)
            outputs.append(self.down_fuse_s3_18_72(self.down_fuse_s3_18(o_18)) + self.down_fuse_s3_36_72(o_36))

            # outputs of each HRModule (except 4th) of stage 3: [fused 36 and 72 feature maps to 18 feature map,
            #                                                   fused 18 and 72 feature maps to 36 feature map,
            #                                                   fused 18 and 36 feature maps to 72 feature map]

            # last (4th HRmodule) of 3th stage should also fuse feature maps to 144 channels
            if self.transition_stage:
                # 18 to 144
                o_18_to_144 = self.transition1_s3_72_144(self.transition1_s3_36_72(self.transition_s3_18_36(o_18)))
                # 36 to 144
                o_36_to_144 = self.transition2_s3_72_144(self.transition2_s3_36_72(o_36))
                # 72 to 144
                o_72_to_144 = self.transition3_s3_72_144(o_72)
                o_144 = o_18_to_144 + o_36_to_144 + o_72_to_144
                outputs.append(o_144)

                # outputs of last (4th) HRmodule of stage 3 : [fused 36 and 72 feature maps to 18 feature map,
                #                                             fused 18 and 72 feature maps to 36 feature map,
                #                                             fused 18 and 36 feature maps to 72 feature map,
                #                                             fused 18, 36 and 72 feature maps and transited to 144 feature map]
        if self.stage == 4:
            # at this stage, 3 subsequent HRModules should be initialized in model definition
            # x[0] = fused 36 and 72 feature maps to 18 feature map
            o = self.s4_b1_block1(x[0])
            o = self.s4_b1_block2(o)
            o = self.s4_b1_block3(o)
            o_18 = self.s4_b1_block4(o)

            # x[1] = fused 18 and 72 feature maps to 36 feature map
            o = self.s4_b2_block1(x[1])
            o = self.s4_b2_block2(o)
            o = self.s4_b2_block3(o)
            o_36 = self.s4_b2_block4(o)

            # x[2] = fused 18 and 36 feature maps to 72 feature map
            o = self.s4_b3_block1(x[2])
            o = self.s4_b3_block2(o)
            o = self.s4_b3_block3(o)
            o_72 = self.s4_b3_block4(o)

            # x[3] = fused 18 and 36 feature maps to 72 feature map
            o = self.s4_b4_block1(x[3])
            o = self.s4_b4_block2(o)
            o = self.s4_b4_block3(o)
            o_144 = self.s4_b4_block4(o)

            # fused 18, 36, 72 and 144 feature maps into 18 feature map
            outputs.append(
                self.up_fuse_s4_36_18(o_36)
                + self.up_fuse_s4_72_18(self.up_fuse_s4_72_72(o_72))
                + self.up_fuse_s4_144_18(self.up2_fuse_s4_144_144(self.up1_fuse_s4_144_144(o_144)))
                + o_18
            )

            # fused 18, 36, 72 and 144 feature maps into 36 feature map
            outputs.append(
                self.down_fuse_s4_18_36(o_18)
                + self.up_fuse_s4_72_36(o_72)
                + self.up_fuse_s4_144_36(self.up3_fuse_s4_144_144(o_144))
                + o_36
            )

            # fused 18, 36, 72 and 144 feature maps into 72 feature map
            outputs.append(
                self.down_fuse_s4_18_72(self.down1_fuse_s4_18(o_18))
                + self.down_fuse_s4_36_72(o_36)
                + self.up_fuse_s4_144_72(o_144)
                + o_72
            )

            # fused 18, 36, 72 and 144 feature maps into 144 feature map
            outputs.append(
                self.down2_fuse_s4_18_144(self.down1_fuse_s4_18_144(self.down2_fuse_s4_18(o_18)))
                + self.down_fuse_s4_36_144(self.down2_fuse_s4_36(o_36))
                + self.up_fuse_s4_72_144(o_72)
                + o_144
            )
            # outputs of each HRmodule of stage 4: [fused 18,36,72 and 144 feature maps into 18 feature map,
            #                                      fused 18,36,72 and 144 feature maps into 36 feature map,
            #                                      fused 18,36,72 and 144 feature maps into 72 feature map,
            #                                      fused 18,36,72 and 144 feature maps into 144 feature map]

        return outputs


class HRNetv2Head(torch.nn.Module):
    """
    Build HRNetv2 representation head to output mask prediction
    """

    def __init__(self, config):
        super().__init__()
        activation_block = get_activation_block(config.activation)

        # upsample 36 to 18 spatial size (1/4 of original image),
        # without changing channels count
        self.upsample_36 = Fusion(36, 36, upsample=True, activation_block=activation_block)

        # upsample 72 to 18 spatial size (1/4 of original image),
        # without changing channels count
        self.upsample1_72 = Fusion(72, 72, upsample=True, activation_block=activation_block)
        self.upsample2_72 = Fusion(72, 72, upsample=True, activation_block=activation_block)

        # upsample 144 to 18 spatial size (1/4 of original image),
        # without changing channels count
        self.upsample1_144 = Fusion(144, 144, upsample=True, activation_block=activation_block)
        self.upsample2_144 = Fusion(144, 144, upsample=True, activation_block=activation_block)
        self.upsample3_144 = Fusion(144, 144, upsample=True, activation_block=activation_block)

        # after feature maps concatenation channels should be 270 (15C, while C=18)
        # self.upsample1_output = Upsample(scale_factor=2)
        # self.upsample2_output = Upsample(scale_factor=2)
        # self.upsample1_output = ConvTranspose2d(270, 270, kernel_size=(2, 2), stride=(2, 2))
        # self.bn1 = BatchNorm3d(270, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv = Conv3d(270, 270, kernel_size=1, stride=1, bias=False)
        # self.bn = BatchNorm3d(270, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.activate = activation_block()
        # predictions: mask([0]) and borders([1])
        # self.conv_seg = Conv3d(270, 2, kernel_size=1, stride=1)

    def forward(self, x):
        # x[0] is 18 feature map (1/4 spatial size of original image)
        # upsample 36 to 18 spatial size (1/4 of original image),
        # without changing channels count
        u_36 = self.upsample_36(x[1])
        # upsample 72 to 18 spatial size (1/4 of original image),
        # without changing channels count
        u_72 = self.upsample2_72(self.upsample1_72(x[2]))
        # upsample 144 to 18 spatial size (1/4 of original image),
        # without changing channels count
        u_144 = self.upsample3_144(self.upsample2_144(self.upsample1_144(x[3])))
        # after feature maps concatenation channels should be 270 (15C, while C=18)
        o = torch.cat([x[0], u_36, u_72, u_144], dim=1)

        # o = self.upsample1_output(o)
        # o = self.bn1(o)
        # o = self.activate(o)
        # o = self.conv(o)
        # o = self.bn(o)
        # o = self.activate(o)
        # # output mask with shape 2xHxW (H and W - original input image height and width)
        # # 2 channels - mask and borders
        # o = self.conv_seg(o)

        return o


class HRNetv2(torch.nn.Module):
    """
    Build HRNetv2 backbone
    """

    def __init__(self, config: HRNetConfig):
        super().__init__()
        activation_module = get_activation_block(config.activation)

        # input stem to downscale image to 1/4 of original size
        self.stem = nn.Sequential(
            Conv3d(config.in_channels, config.stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm3d(config.stem_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation_module(),
            Conv3d(config.stem_channels, config.stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm3d(config.stem_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation_module(),
        )

        # stage 1 containing 1 HRModule
        self.stage_1 = HRModule(config, stage=1)

        # stage 2 containing 1 HRModule
        self.stage_2 = HRModule(config, stage=2)

        # stage 3 containing 4 HRModule's
        self.stage_3_1 = HRModule(config, stage=3)
        self.stage_3_2 = HRModule(config, stage=3)
        self.stage_3_3 = HRModule(config, stage=3)
        # HRModule with transition stage (transiting 144 feature maps to stage 4)
        self.stage_3_4 = HRModule(config, stage=3, transition_stage=True)

        # stage 4 containing 3 HRModule's
        self.stage_4_1 = HRModule(config, stage=4)
        self.stage_4_2 = HRModule(config, stage=4)
        self.stage_4_3 = HRModule(config, stage=4)

        self.decode_head = HRNetv2Head(config)

        self.apply(initialize_weights)

    def forward(self, x):
        o = self.stem(x)

        # append output feature map of stem to list,
        # because HRModule works with list of values
        input_list = []
        input_list.append(o)

        outputs_list = self.stage_1(input_list)
        # outputs of stage 1: [18 channels feature map,
        #                       36 channels feature map]

        outputs_list = self.stage_2(outputs_list)
        # outputs of stage 2: [fused 18 channels feature map,
        #                       fused 36 channels feature map,
        #                       fused(transited 18 to 72 channels feature map
        #                       + transited 36 to 72 channels feature map)]

        outputs_list = self.stage_3_1(outputs_list)
        outputs_list = self.stage_3_2(outputs_list)
        outputs_list = self.stage_3_3(outputs_list)
        # outputs of each HRModule (except 4th) of stage 3: [fused 36 and 72 feature maps to 18 feature map,
        #                                                   fused 18 and 72 feature maps to 36 feature map,
        #                                                   fused 18 and 36 feature maps to 72 feature map]
        outputs_list = self.stage_3_4(outputs_list)
        # outputs of last (4th) HRmodule of stage 3 : [fused 36 and 72 feature maps to 18 feature map,
        #                                             fused 18 and 72 feature maps to 36 feature map,
        #                                             fused 18 and 36 feature maps to 72 feature map,
        #                                             fused 18, 36 and 72 feature maps and transited to 144 feature map]

        outputs_list = self.stage_4_1(outputs_list)
        outputs_list = self.stage_4_2(outputs_list)
        outputs_list = self.stage_4_3(outputs_list)
        # outputs of each HRmodule of stage 4: [fused 18,36,72 and 144 feature maps into 18 feature map,
        #                                      fused 18,36,72 and 144 feature maps into 36 feature map,
        #                                      fused 18,36,72 and 144 feature maps into 72 feature map,
        #                                      fused 18,36,72 and 144 feature maps into 144 feature map]

        o = self.decode_head(outputs_list)

        return o


import einops


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        r = self.upscale_factor
        b, c, d, h, w = x.shape

        out = einops.rearrange(x, "b (c r1 r2 r3) d h w -> b c (d r1) (h r2) (w r3)", r1=r, r2=r, r3=r)
        return out


class HRNetv2ForObjectDetection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = HRNetv2(config)
        self.up = nn.Sequential(nn.Conv3d(270, 64 * 8, kernel_size=1), PixelShuffle3d(2))
        self.head = ObjectDetectionHead(
            in_channels=64, num_classes=5, intermediate_channels=48, offset_intermediate_channels=16, stride=2
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        x = self.backbone(volume)
        x_up = self.up(x)
        logits, offsets = self.head(x_up)

        logits = [logits]
        offsets = [offsets]
        strides = [self.head.stride]

        if torch.jit.is_tracing():
            return logits, offsets

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = object_detection_loss(logits, offsets, strides, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, strides=strides, loss=loss, loss_dict=loss_dict)


if __name__ == "__main__":
    config = HRNetConfig()
    model = HRNetv2ForObjectDetection(config)
    print(count_parameters(model, human_friendly=True))
    x = torch.randn(1, 1, 96, 256, 256)
    with torch.no_grad():
        y = model(x)
        print(y)
