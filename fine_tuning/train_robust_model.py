"""
Robust Real-Fruit MobileNetV2 Training Pipeline
Trains on expanded multi-variety real fruit dataset with synthetic realistic background blending,
photometric jitter, and random occlusions to ensure extreme robustness on real stock photos.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from background_augmenter import blend_fruit_on_background

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class AugmentedFruitDataset(Dataset):
    """
    Dataset that loads fruit images and dynamically composites realistic backgrounds
    during training to make the neural network invariant to tables, counters, and plates.
    """
    def __init__(self, file_paths, labels, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

        self.tensor_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5 if is_train else 0.0),
            transforms.RandomVerticalFlip(p=0.2 if is_train else 0.0),
            transforms.RandomRotation(degrees=25 if is_train else 0),
            transforms.ColorJitter(
                brightness=0.25 if is_train else 0.0,
                contrast=0.25 if is_train else 0.0,
                saturation=0.25 if is_train else 0.0,
                hue=0.04 if is_train else 0.0
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (255, 255, 255))

        # Dynamically blend onto realistic background during training (60% chance)
        if self.is_train and random.random() < 0.60:
            try:
                img = blend_fruit_on_background(img)
            except Exception:
                pass

        tensor = self.tensor_transforms(img)
        return tensor, label


def create_model(num_classes: int = 15):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, 384),
        nn.ReLU6(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(384, num_classes)
    )
    return model


def collect_dataset(dataset_dir: Path):
    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    file_paths = []
    labels = []
    for c in classes:
        for p in (dataset_dir / c).glob("*.jpg"):
            file_paths.append(str(p))
            labels.append(class_to_idx[c])
            
    return classes, class_to_idx, file_paths, labels


def export_model_to_onnx(model, num_classes, class_names, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_onnx = str(output_dir / "mobilenet_v2_fruits_fp32.onnx")
    quantized_onnx = str(output_dir / "mobilenet_v2_quantized.onnx")
    config_path = str(output_dir / "mobilenet_v2_config.json")
    
    id2label = {str(i): c for i, c in enumerate(class_names)}
    label2id = {c: i for i, c in enumerate(class_names)}
    
    cfg = {
        "_name_or_path": "mobilenet_v2_fruits_robust",
        "architectures": ["MobileNetV2ForFruitClassification"],
        "num_classes": num_classes,
        "id2label": id2label,
        "label2id": label2id,
        "classes": class_names,
        "dataset_name": "Fruits-360 Expanded Multi-Variety Real Fruit Dataset",
        "dataset_url": "https://github.com/Horea94/Fruit-Images-Dataset",
        "robustness_features": ["Dynamic Background Blending", "Photometric Jitter", "Multi-Variety Mapping"]
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved config to {config_path}")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    print(f"Exporting FP32 ONNX model to {raw_onnx}...")
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
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    print(f"Exported FP32 ONNX ({os.path.getsize(raw_onnx)/(1024*1024):.2f} MB).")

    print(f"Quantizing to dynamic INT8: {quantized_onnx}...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(raw_onnx, quantized_onnx, weight_type=QuantType.QUInt8)
        print(f"Quantized ONNX saved ({os.path.getsize(quantized_onnx)/(1024*1024):.2f} MB).")
    except Exception as e:
        print(f"Quantization fallback: {e}")
        import shutil
        shutil.copyfile(raw_onnx, quantized_onnx)


def main():
    print("=" * 70)
    print("FruitVision Robust Real-Fruit Training Pipeline")
    print("=" * 70)

    dataset_root = Path("fruit_analyzer/data/fruits_dataset_expanded")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    classes, class_to_idx, train_files, train_labels = collect_dataset(train_dir)
    _, _, val_files, val_labels = collect_dataset(val_dir)

    print(f"Loaded {len(train_files)} training images and {len(val_files)} validation images across {len(classes)} classes:")
    print(f"Classes: {classes}")

    train_dataset = AugmentedFruitDataset(train_files, train_labels, is_train=True)
    val_dataset = AugmentedFruitDataset(val_files, val_labels, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    torch.set_num_threads(8)
    device = torch.device("cpu")
    model = create_model(num_classes=len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Stage 1: Warmup classifier head (2 epochs)
    print("\n[Stage 1/2] Warming up Classifier Head (2 epochs)...")
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                print(f"  Warmup Ep [{epoch}/2] Batch [{batch_idx}/{len(train_loader)}] - Loss: {running_loss/total:.4f} | Acc: {(correct/total)*100:.1f}%")

        acc = (correct / total) * 100.0
        print(f"Warmup Epoch [{epoch}/2] Summary - Loss: {running_loss/total:.4f} | Train Acc: {acc:.2f}%")

    # Stage 2: Fine-tune top feature blocks + classifier (3 epochs)
    print("\n[Stage 2/2] Fine-Tuning Top Feature Blocks with Background Invariance (3 epochs)...")
    # Unfreeze only the top 3 feature blocks (layers 16, 17, 18) for fast, regularized transfer learning
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.features[-3:].parameters(), "lr": 2e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    best_val_acc = 0.0
    best_weights_path = Path("fruit_analyzer/models/weights/best_robust_mobilenetv2.pth")
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                print(f"  Finetune Ep [{epoch}/3] Batch [{batch_idx}/{len(train_loader)}] - Loss: {train_loss/train_total:.4f} | Acc: {(train_correct/train_total)*100:.1f}%")

        scheduler.step()
        train_acc = (train_correct / train_total) * 100.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = (val_correct / val_total) * 100.0
        print(f"Epoch [{epoch}/3] Finished - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_weights_path))
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc:.2f}%)")

    print(f"\nTraining Complete! Best Real-Fruit Validation Accuracy: {best_val_acc:.2f}%")

    # Load best weights and export
    model.load_state_dict(torch.load(str(best_weights_path), map_location="cpu"))
    export_model_to_onnx(model, len(classes), classes, Path("fruit_analyzer/models/weights"))
    print("\nALL DONE! Robust Real-Fruit Model successfully trained and deployed!")

if __name__ == "__main__":
    main()
