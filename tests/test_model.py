import numpy as np
import timm.models.maxxvit
import torch
from pytorch_toolbelt.utils import count_parameters
from torch import Tensor

from cryoet.data.functional import normalize_volume_to_unit_range
from cryoet.ensembling import model_from_checkpoint
from cryoet.modelling.ch_models.adaptors import MdlCh20dCe2_resnet34, MdlCh20dCe2c2_resnet34, MdlCh20dCe2_effnetb3
from cryoet.modelling.detection.dynunet import DynUNetForObjectDetectionConfig, DynUNetForObjectDetection
from cryoet.modelling.detection.functional import convert_2d_to_3d, gaussian_blur_3d
from cryoet.modelling.detection.maxvit_unet25d import MaxVitUnet25d, MaxVitUnet25dConfig
from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionV2Config,
    SegResNetForObjectDetectionV2,
)
from cryoet.modelling.detection.task_aligned_assigner import check_points_inside_bboxes
from cryoet.modelling.detection.unet3d_detection import UNet3DForObjectDetectionConfig, UNet3DForObjectDetection

from cryoet.modelling.detection.unetr import SwinUNETRForObjectDetectionConfig, SwinUNETRForObjectDetection


@torch.no_grad()
def test_MdlCh20dCe2_resnet34():
    input = torch.randn((1, 1, 96, 96, 96)).cuda().half()
    m1 = MdlCh20dCe2_resnet34().eval().cuda().half()
    output = m1(input)
    assert output[0].shape == (1, 6, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)
    assert output[1].shape == (1, 3, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)

    traced_model = torch.jit.trace(m1, input)
    output = traced_model(input)
    assert output[0].shape == (1, 6, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)
    assert output[1].shape == (1, 3, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)

    torch.onnx.export(m1, input, "mdl_ch_20d_ce2_resnet34.onnx")


@torch.no_grad()
def test_MdlCh20dCe2c2_resnet34():
    input = torch.randn((1, 1, 96, 96, 96)).cuda().half()
    m2 = MdlCh20dCe2c2_resnet34().eval().cuda().half()
    output = m2(input)
    assert output[0].shape == (1, 6, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)
    assert output[1].shape == (1, 3, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)

    traced_model = torch.jit.trace(m2, input)
    output = traced_model(input)
    assert output[0].shape == (1, 6, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)
    assert output[1].shape == (1, 3, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)

    torch.onnx.export(m2, input, "mdl_ch_20d_ce2c2_resnet34.onnx")


@torch.no_grad()
def test_MdlCh20dCe2_effnetb3():
    input = torch.randn((1, 1, 96, 96, 96)).cuda().half()
    m3 = MdlCh20dCe2_effnetb3().eval().cuda().half()
    output = m3(input)
    print(output[0].shape, output[1].shape)
    assert output[0].shape == (1, 6, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)
    assert output[1].shape == (1, 3, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)

    traced_model = torch.jit.trace(m3, input)
    output = traced_model(input)
    assert output[0].shape == (1, 6, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)
    assert output[1].shape == (1, 3, input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)

    torch.onnx.export(m3, input, "mdl_ch_20d_ce2_effnetb3.onnx")


#
# def test_check_points_inside_bboxes():
#     points = torch.arange(20).float()
#     points = torch.stack([points, torch.zeros_like(points), torch.zeros_like(points)], dim=-1)
#     points = points.unsqueeze(0)  # [1, 20, 3]
#
#     true_centers = torch.tensor([0, 0, 0]).view(1, 1, 3).repeat(1, 5, 1).float()  # Center
#
#     true_sigmas = torch.tensor([6, 9, 15, 13, 13.5]).float().view(1, -1, 1)  # Radius
#
#     mask = check_points_inside_bboxes(points, true_centers, true_sigmas)
#     print(mask)
#
#
# def test_monai_transforms():
#     import monai.transforms as mt
#
#     data = torch.randn(1, 6, 128, 128, 128)
#
#     def mean_std_renormalization(volume: Tensor):
#         """
#         Renormalize the volume to have zero mean and unit variance.
#         :param volume: Tensor of shape (B, C, D, H, W)
#         """
#         mean = volume.mean(dim=(1, 2, 3, 4), keepdim=True)
#         std = volume.std(dim=(1, 2, 3, 4), keepdim=True)
#         volume = (volume - mean) / std
#         return volume
#
#     data_minmax = normalize_volume_to_unit_range(data)
#     data_meanstd = mean_std_renormalization(data_minmax)
#
#     output = mt.NormalizeIntensity()(data)
#
#     mean_diff = torch.abs(data_meanstd - output).mean()
#     print(mean_diff.item())
#
#     max_diff = torch.abs(data_meanstd - output).max()
#     print(max_diff.item())
#
#
# def test_gaussian_blur_3d():
#
#     data = torch.randn(1, 6, 128, 128, 128)
#     data_blur = gaussian_blur_3d(data, 3, 1)
#     print(data_blur.shape)
#
#
# def test_convertmaxvit():
#     model = timm.create_model("maxxvit_rmlp_nano_rw_256.sw_in1k", in_channels=1, pretrained=True, features_only=True)
#
#     model3d = convert_2d_to_3d(model)
#
#     x = torch.randn(2, 1, 96, 96, 96)
#     out = model3d(x)
#     print(out)
#
#
# def test_segresnetv2():
#     config = SegResNetForObjectDetectionV2Config(window_size=96)
#     model = SegResNetForObjectDetectionV2(config)
#     x = torch.randn(2, 1, 96, 96, 96)
#     out = model(x)
#     print(out)
#
#
# #
# # def test_unet_3d():
# #
# #     # Example usage with decode_final_stride=2 -> output is half resolution
# #     config = UNet3DForObjectDetectionConfig()
# #     model = UNet3DForObjectDetection(config)
# #
# #     # Fake input: batch_size=2, channels=1, depth=64, height=64, width=64
# #     x = torch.randn(2, 1, 96, 96, 96)
# #     out = model(x)
# #
# #     print("Model summary:", count_parameters(model, human_friendly=True))
# #     print("Input shape:", x.shape)
# #     print("Output shape:", [x.shape for x in out.logits])
#
#
# # def test_maxvitunet25d():
# #     config = MaxVitUnet25dConfig(img_size=128)
# #     model = MaxVitUnet25d(config)
# #
# #     x = torch.randn(2, 1, 32, config.img_size, config.img_size)
# #     out = model(x)
# #
# #     print("Model summary:", count_parameters(model, human_friendly=True))
# #     print("Input shape:", x.shape)
# #     print("Output shape:", [x.shape for x in out.logits])
#
#
# def test_dynunet():
#     config = DynUNetForObjectDetectionConfig()
#     model = DynUNetForObjectDetection(config)
#
#     x = torch.randn(2, 1, 96, 96, 96)
#     out = model(x)
#     print("Model summary:", count_parameters(model, human_friendly=True))
#     print("Input shape:", x.shape)
#     # print("Output shape:", [x.shape for x in out.logits])
#
#
# def test_dynunet2():
#     config = DynUNetForObjectDetectionConfig(object_size=32)
#     model = DynUNetForObjectDetection(config)
#
#     x = torch.randn(2, 1, 96, 96, 96)
#     out = model(x)
#     print("Model summary:", count_parameters(model, human_friendly=True))
#     print("Input shape:", x.shape)
#     # print("Output shape:", [x.shape for x in out.logits])
#
#
# def test_unetr():
#     config = SwinUNETRForObjectDetectionConfig()
#     model = SwinUNETRForObjectDetection(config)
#
#     x = torch.randn(2, 1, 96, 96, 96)
#     out = model(x)
#     print("Model summary:", count_parameters(model, human_friendly=True))
#     print("Input shape:", x.shape)
#     # print("Output shape:", [x.shape for x in out.logits])
