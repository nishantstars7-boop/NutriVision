import os
import sys
import json
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

def main():
    weights_dir = Path("fruit_analyzer/models/weights")
    ckpt_path = weights_dir / "best_mobilenetv3_stock.pth"
    config_path = weights_dir / "mobilenet_v3_config.json"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    num_classes = config["num_classes"]
    print(f"Loading checkpoint {ckpt_path} with {num_classes} classes...")
    
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1024, num_classes)
    )
    
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    raw_onnx = weights_dir / "mobilenet_v3_fruits_fp32.onnx"
    quant_onnx = weights_dir / "mobilenet_v3_quantized.onnx"
    
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    
    print(f"Exporting FP32 ONNX using classic TorchScript exporter (dynamo=False)...")
    torch.onnx.export(
        model,
        dummy_input,
        str(raw_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"FP32 ONNX exported: {raw_onnx.stat().st_size / (1024*1024):.2f} MB")
    
    # Verify FP32 with ONNX Runtime
    import onnxruntime as ort
    session_fp32 = ort.InferenceSession(str(raw_onnx), providers=["CPUExecutionProvider"])
    inp_name = session_fp32.get_inputs()[0].name
    out = session_fp32.run(None, {inp_name: dummy_input.numpy()})
    print(f"FP32 output shape: {out[0].shape}")
    
    # Quantize to dynamic INT8
    print(f"Quantizing to dynamic INT8...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=str(raw_onnx),
            model_output=str(quant_onnx),
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized INT8 ONNX saved: {quant_onnx.stat().st_size / (1024*1024):.2f} MB")
        session_quant = ort.InferenceSession(str(quant_onnx), providers=["CPUExecutionProvider"])
        out_q = session_quant.run(None, {inp_name: dummy_input.numpy()})
        print(f"INT8 output shape: {out_q[0].shape}")
    except Exception as e:
        print(f"Quantization notice: {e}, using FP32 as production model.")
        shutil.copyfile(raw_onnx, quant_onnx)
        
    print("\nSUCCESS! MobileNetV3 model exported and ready for production inference!")

if __name__ == "__main__":
    main()
