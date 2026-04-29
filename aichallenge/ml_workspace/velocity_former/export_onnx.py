"""Export a trained VelocityFormer checkpoint to ONNX.

Mirrors `tiny_lidar_net/convert_weight.py` in spirit, but emits an ONNX file
because the runtime path uses ONNX Runtime (BERT is too involved for the
pure-NumPy approach used by tiny_lidar_net).
"""

import argparse
from pathlib import Path

import torch

from lib.model import VelocityFormer


def export(
    pretrained_model: str,
    input_size: int,
    num_labels: int,
    ckpt_path: Path,
    output_path: Path,
    opset: int = 14,
) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = VelocityFormer(
        pretrained_model=pretrained_model,
        input_size=input_size,
        num_labels=num_labels,
        load_pretrained_weights=False,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 入力IDは [0, 360) のlong
    dummy_input = torch.zeros(1, input_size, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path.as_posix(),
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported ONNX model to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export VelocityFormer PyTorch ckpt to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Source .pth checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx path.")
    parser.add_argument("--pretrained-model", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--input-size", type=int, default=12)
    parser.add_argument("--num-labels", type=int, default=1)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export(
        args.pretrained_model,
        args.input_size,
        args.num_labels,
        args.ckpt,
        args.output,
        args.opset,
    )


if __name__ == "__main__":
    main()
