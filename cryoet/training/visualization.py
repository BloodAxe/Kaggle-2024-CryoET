import cv2
import numpy as np
import torch
from torch import Tensor
from pytorch_toolbelt.utils.visualization import vstack_header, hstack_autopad, vstack_autopad


def pseudo_colorize(hm):
    img = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return img[..., ::-1]


def render_heatmap(heatmap: Tensor):
    """
    :param heatmap: Tensor of shape (C, D, H, W)
    """
    heatmap = heatmap.float().cpu().numpy()

    # Make 3 slices in HW, DH, DW planes (middle slices)
    slice_hw = heatmap[:, heatmap.shape[1] // 2, :, :]
    slice_dh = heatmap[:, :, heatmap.shape[2] // 2, :]
    slice_dw = heatmap[:, :, :, heatmap.shape[3] // 2]

    slice_hw = hstack_autopad(
        [vstack_header(pseudo_colorize(x), f"min={x.min():.3f} max={x.max():.3f}") for x in slice_hw], spacing=2
    )
    slice_dh = hstack_autopad(
        [vstack_header(pseudo_colorize(x), f"min={x.min():.3f} max={x.max():.3f}") for x in slice_dh], spacing=2
    )
    slice_dw = hstack_autopad(
        [vstack_header(pseudo_colorize(x), f"min={x.min():.3f} max={x.max():.3f}") for x in slice_dw], spacing=2
    )

    return vstack_autopad(
        [vstack_header(slice_hw, "HW Slice"), vstack_header(slice_dh, "DH Slice"), vstack_header(slice_dw, "DW Slice")]
    )


if __name__ == "__main__":
    hm = torch.randn(5, 180, 640, 640).sigmoid()
    image = render_heatmap(hm)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(image)
    plt.tight_layout()
    plt.show()
