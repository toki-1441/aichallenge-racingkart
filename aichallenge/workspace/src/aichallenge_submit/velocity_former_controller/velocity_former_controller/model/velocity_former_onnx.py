"""ONNX Runtime wrapper for VelocityFormer inference."""

from typing import Optional

import numpy as np


class VelocityFormerOnnxRunner:
    """Loads a VelocityFormer ONNX model and runs single-sample inference.

    The model is expected to take a single int64 input named ``input_ids`` of
    shape (B, input_size) and produce a float32 output of shape (B, num_labels).
    """

    def __init__(self, onnx_path: str, providers: Optional[list] = None):
        # 遅延 import: onnxruntime が無い環境でも import 自体は通したい
        import onnxruntime as ort

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, input_ids: np.ndarray) -> np.ndarray:
        """Run inference on a single batch.

        Args:
            input_ids: int64 array of shape (B, seq_len) or (seq_len,).

        Returns:
            np.ndarray of shape (B, num_labels) or (num_labels,).
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.astype(np.int64, copy=False)

        outputs = self.session.run([self.output_name], {self.input_name: input_ids})
        return outputs[0]
