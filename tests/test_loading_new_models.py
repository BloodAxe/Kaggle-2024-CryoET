import torch

from cryoet.ensembling import model_from_checkpoint


def test_load_weights_cryo_cfg_ch_48h_ce2c2():
    model = model_from_checkpoint(
        "../models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed151584.pth"
    )
    data = torch.randn(1, 1, 192, 128, 128)

    outputs = model(data)
    assert len(outputs) == 2
    print(outputs[0].shape, outputs[1].shape)

    assert outputs[0].shape == (1, 6, 192 // 2, 128 // 2, 128 // 2)
    assert outputs[1].shape == (1, 3, 192 // 2, 128 // 2, 128 // 2)


def test_load_weights_cryo_cfg_ch_48j2():
    model = model_from_checkpoint("../models/weights-cryo-cfg-ch-48j2/models/cfg_ch_48j2/fold-1/checkpoint_last_seed289559.pth")
    data = torch.randn(1, 1, 192, 128, 128)

    outputs = model(data)
    assert len(outputs) == 2
    print(outputs[0].shape, outputs[1].shape)
    assert outputs[0].shape == (1, 6, 192 // 2, 128 // 2, 128 // 2)
    assert outputs[1].shape == (1, 3, 192 // 2, 128 // 2, 128 // 2)


def test_load_weights_cryo_cfg_ch_48k():
    model = model_from_checkpoint("../models/weights-cryo-cfg-ch-48k/models/cfg_ch_48k/fold-1/checkpoint_last_seed32073.pth")
    data = torch.randn(1, 1, 192, 128, 128)

    outputs = model(data)
    assert len(outputs) == 2
    print(outputs[0].shape, outputs[1].shape)
    assert outputs[0].shape == (1, 6, 192 // 2, 128 // 2, 128 // 2)
    assert outputs[1].shape == (1, 3, 192 // 2, 128 // 2, 128 // 2)
