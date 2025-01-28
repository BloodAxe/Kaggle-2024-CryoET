import shutil
import numpy as np
import onnxruntime as ort

from pathlib import Path
from tqdm import tqdm


def compile_onnx_to_trt(onnx_path, batch_size=1):
    shutil.copy(onnx_path, "./")
    onnx_filename = str(Path(onnx_path).name)
    args = {"volume": np.random.randn(batch_size, 1, 192, 128, 128).astype(np.float16)}

    sess_options = ort.SessionOptions()
    session1 = ort.InferenceSession(
        path_or_bytes=onnx_filename,
        providers=[
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_max_workspace_size": 12 * 1073741824,
                    "trt_builder_optimization_level": 4,
                    "trt_timing_cache_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_timing_cache_path": "/kaggle/working/v4_segresnet_dynunet/trt_timing_cache",
                    "trt_engine_cache_path": "/kaggle/working/v4_segresnet_dynunet/trt_engine_cache",
                    "trt_dump_ep_context_model": True,
                    "trt_ep_context_file_path": "./",
                },
            )
        ],
        sess_options=sess_options,
    )

    for _ in tqdm(range(80 // batch_size)):
        session1.run(["scores", "offsets"], args)
