import torch
from pytorch_toolbelt.utils import count_parameters

from cryoet.modelling.detection.dynunet import DynUNetForObjectDetectionConfig, DynUNetForObjectDetection
from cryoet.modelling.detection.maxvit_unet25d import MaxVitUnet25d, MaxVitUnet25dConfig
from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionConfig,
    SegResNetForObjectDetectionV2,
)
from cryoet.modelling.detection.unet3d_detection import UNet3DForObjectDetectionConfig, UNet3DForObjectDetection
from cryoet.modelling.unetr_point_detection import SwinUNETRForPointDetection


def test_model_forward():
    config = SwinUNETRForPointDetectionConfig()
    model = SwinUNETRForPointDetection(config)

    input = torch.randn((1, 1, 96, 96, 96))
    labels = torch.randn((1, config.num_classes, 96, 96, 96))
    labels = (labels > 0.5).float()
    output = model(input, labels=labels)

    print(output.logits.shape)
    print(output.loss)


def test_segresnetv2():
    config = SegResNetForObjectDetectionConfig(window_size=96)
    model = SegResNetForObjectDetectionV2(config)
    x = torch.randn(2, 1, 96, 96, 96)
    out = model(x)
    print(out)


def test_unet_3d():

    # Example usage with decode_final_stride=2 -> output is half resolution
    config = UNet3DForObjectDetectionConfig()
    model = UNet3DForObjectDetection(config)

    # Fake input: batch_size=2, channels=1, depth=64, height=64, width=64
    x = torch.randn(2, 1, 96, 96, 96)
    out = model(x)

    print("Model summary:", count_parameters(model, human_friendly=True))
    print("Input shape:", x.shape)
    print("Output shape:", [x.shape for x in out.logits])


def test_maxvitunet25d():
    config = MaxVitUnet25dConfig(img_size=128)
    model = MaxVitUnet25d(config)

    x = torch.randn(2, 1, 32, config.img_size, config.img_size)
    out = model(x)

    print("Model summary:", count_parameters(model, human_friendly=True))
    print("Input shape:", x.shape)
    print("Output shape:", [x.shape for x in out.logits])


def test_dynunet():
    config = DynUNetForObjectDetectionConfig()
    model = DynUNetForObjectDetection(config)

    x = torch.randn(2, 1, 96, 96, 96)
    out = model(x)
    print("Model summary:", count_parameters(model, human_friendly=True))
    print("Input shape:", x.shape)
    # print("Output shape:", [x.shape for x in out.logits])
