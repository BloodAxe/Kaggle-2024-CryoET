from torch.utils.data import Dataset

from cryoet.data.functional import normalize_volume_to_unit_range
from cryoet.data.parsers import (
    get_volume_and_objects,
    TARGET_CLASSES,
    ANGSTROMS_IN_PIXEL,
    AnnotatedVolume,
)


class CryoETObjectDetectionDataset(Dataset):
    def __init__(self, sample: AnnotatedVolume):

        self.sample = sample

        self.study = sample.study
        self.split = sample.split
        self.mode = sample.mode
        self.volume_data = sample.volume
        self.volume_shape = sample.volume.shape
        self.object_centers = sample.centers
        self.object_labels = sample.labels
        self.object_radii = sample.radius

        self.object_centers_px = sample.centers_px
        self.object_radii_px = sample.radius_px

        self.num_classes = len(TARGET_CLASSES)
