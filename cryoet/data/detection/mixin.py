import torch


class ObjectDetectionMixin:
    def convert_to_dict(
        self,
        volume,
        centers,
        labels,
        radii,
        study_name,
        volume_shape,
        tile_offsets_zyx,
        mode,
    ):
        volume = torch.from_numpy(volume).unsqueeze(0).float()
        labels = torch.cat(
            [
                torch.from_numpy(centers).float(),
                torch.from_numpy(labels[:, None]).float(),
                torch.from_numpy(radii[:, None]).float(),
            ],
            dim=-1,
        )  # N x 5

        return {
            "volume": volume,
            "labels": labels,
            "tile_offsets_zyx": torch.tensor(tile_offsets_zyx),
            "volume_shape": torch.tensor(volume_shape),
            "study": study_name,
            "mode": mode,
        }
