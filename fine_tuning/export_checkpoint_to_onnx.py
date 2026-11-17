import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import json
import torch
import torchvision.models as models
from pathlib import Path

def export():
    weights_dir = Path("fruit_analyzer/models/weights")
    ckpt_path = weights_dir / "best_fruits360_mobilenetv2.pth"
    config_path = weights_dir / "mobilenet_v2_config.json"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    num_classes = config["num_classes"]
    
    print(f"Loading checkpoint {ckpt_path} with {num_classes} classes...")
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features, 384),
        torch.nn.ReLU6(inplace=True),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(384, num_classes)
    )
    
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    raw_onnx = str(weights_dir / "mobilenet_v2_fruits_fp32.onnx")
    quantized_onnx = str(weights_dir / "mobilenet_v2_quantized.onnx")
    
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    
    print(f"Exporting to ONNX: {raw_onnx}...")
    torch.onnx.export(
        model,
        dummy_input,
        raw_onnx,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"Exported FP32 ONNX ({os.path.getsize(raw_onnx)/(1024*1024):.2f} MB).")
    
    print(f"Quantizing to INT8: {quantized_onnx}...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            raw_onnx,
            quantized_onnx,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized ONNX saved ({os.path.getsize(quantized_onnx)/(1024*1024):.2f} MB).")
    except Exception as e:
        print(f"Quantization fallback: {e}")
        import shutil
        shutil.copyfile(raw_onnx, quantized_onnx)

    print("ONNX export and quantization complete!")

if __name__ == "__main__":
    export()
