import torch

from cryoet.modelling.configuration import SwinUNETRForPointDetectionConfig
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
