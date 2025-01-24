import numpy as np
import onnxruntime


def main():
    session = onnxruntime.InferenceSession("v4_v6_ensemble_192x128x128_opset_15.onnx", providers=["CUDAExecutionProvider"])
    input_data = np.random.randn(1, 1, 192, 128, 128).astype(np.float32)

    outputs = session.run(None, {"x": input_data})  # or provide a list of output names if you prefer

    print("Output shape:", outputs[0].shape)


if __name__ == "__main__":
    main()
