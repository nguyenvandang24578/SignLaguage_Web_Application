# convert_gru.py
import torch
import numpy as np

device = torch.device("cpu")  # export luôn trên CPU

model = torch.jit.load("./gru.pt", map_location=device)
model.eval()

# Dummy input: (1, 30, 63)
dummy_input = torch.randn(1, 30, 63)

torch.onnx.export(
    model,
    dummy_input,
    "./gru.onnx",
    input_names=["keypoints"],
    output_names=["logits"],
    dynamic_axes={"keypoints": {0: "batch_size"}},
    opset_version=14
)

print("Export ONNX thành công!")