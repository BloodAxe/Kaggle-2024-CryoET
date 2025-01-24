import torch
from fire import Fire
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        offsets = []
        scores = []

        for model in self.models:
            (logits,), (offset,) = model(x)
            scores.append(logits.sigmoid())
            offsets.append(offset)

        scores = torch.mean(torch.stack(scores, dim=0), dim=0)
        offsets = torch.mean(torch.stack(offsets, dim=0), dim=0)
        return scores, offsets


@torch.no_grad()
def main(*traced_models):
    models = [torch.jit.load(p, map_location="cuda") for p in traced_models]
    ensemble = Ensemble(models).eval()

    dummy_input = torch.randn(1, 1, 192, 128, 128, device="cuda")
    ensemble = torch.jit.trace(ensemble, dummy_input)
    torch.onnx.export(
        model=ensemble,
        args=dummy_input,
        f="v4_v6_ensemble_192x128x128_opset_15.onnx",
        verbose=False,
        verify=True,
        opset_version=15,
    )
    # torch.onnx.export(ensemble, torch.randn(1, 1, 128, 128, 128, device="cuda"), "v4_v6_ensemble_128x128x128.onnx", verbose=True, verify=True)


if __name__ == "__main__":
    Fire(main)
