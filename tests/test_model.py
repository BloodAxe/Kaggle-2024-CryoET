import torch


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


def test_unet_3d():
    from pytorch_toolbelt.utils import count_parameters

    # Example usage with decode_final_stride=2 -> output is half resolution
    config = UNet3DForObjectDetectionConfig()
    model = UNet3DForObjectDetection(config)

    # Fake input: batch_size=2, channels=1, depth=64, height=64, width=64
    x = torch.randn(2, 1, 96, 96, 96)
    out = model(x)

    print("Model summary:", count_parameters(model, human_friendly=True))
    print("Input shape:", x.shape)
    print("Output shape:", out.logits.shape)
    # With decode_final_stride=2, we expect an output shape of [2, 2, 32, 32, 32].
