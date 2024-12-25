from transformers import PretrainedConfig


class PointDetectionModelConfig(PretrainedConfig):
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
