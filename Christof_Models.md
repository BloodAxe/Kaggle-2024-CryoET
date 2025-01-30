mdl_ch_20d_ce2 #1
backbone_args = dict(spatial_dims=3,
in_channels=1,
out_channels=6,
backbone=‘resnet34’,                         pretrained=False)
https://www.kaggle.com/code/christofhenkel/weights-cryo-cfg-ch-48j2

mdl_ch_20d_ce2 #2
backbone_args = dict(spatial_dims=3,
in_channels=1,
out_channels=6,
backbone=‘efficientnet-b3’,                         pretrained=False)
https://www.kaggle.com/code/christofhenkel/weights-cryo-cfg-ch-48k

mdl_ch_20d_ce2c2 #1
backbone_args = dict(spatial_dims=3,
in_channels=1,
out_channels=6,
backbone=‘resnet34’,                         pretrained=False)
https://www.kaggle.com/code/christofhenkel/weights-cryo-cfg-ch-48h-ce2c2 (ed

kaggle kernels output christofhenkel/weights-cryo-cfg-ch-48j2      -p models/weights-cryo-cfg-ch-48j2
kaggle kernels output christofhenkel/weights-cryo-cfg-ch-48k       -p models/weights-cryo-cfg-ch-48k
kaggle kernels output christofhenkel/weights-cryo-cfg-ch-48h-ce2c2 -p models/weights-cryo-cfg-ch-48h-ce2c2

trace_model.py \
 models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed151584.pth \
 models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed163583.pth \
 models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed849301.pth \
 models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed899252.pth

evaluate_ensemble.py models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed151584.jit models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed163583.jit models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed849301.jit models/weights-cryo-cfg-ch-48h-ce2c2/models/cfg_ch_48h_ce2c2/fold-1/checkpoint_last_seed899252.jit --output_dir=models/chris_ensemble_eval --validate_on_rot90=False --validate_on_x_flips=False --validate_on_y_flips=False --min_score_threshold=0.01