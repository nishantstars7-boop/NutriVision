"""
Train Production-Grade MobileNetV3-Large on Real Stock Dataset + Fruits-360
Features:
- Pretrained ImageNet-1k weights with Squeeze-and-Excitation (SE) channel attention
- Combined 2,400+ dataset: Genuine in-the-wild stock photos + multi-variety Fruits-360
- Explicit 16th class: 'Background' (empty tables, cutting boards, plates, countertops)
- Heavy photometric & geometric data augmentation + Label Smoothing
- Two-stage progressive fine-tuning (Head warmup -> Top SE blocks fine-tuning)
- Dynamic INT8 ONNX quantization export
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Ensure background augmenter is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent))
from background_augmenter import BackgroundAugmenter

torch.set_num_threads(8)

CLASSES = [
    "Apple", "Avocado", "Banana", "Cherry", "Grape", "Kiwi", "Lemon", "Mango",
    "Orange", "Peach", "Pear", "Pineapple", "Pomegranate", "Strawberry", "Watermelon",
    "Background"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

STOCK_DATASET_DIR = Path("fruit_analyzer/data/real_stock_dataset")
EXPANDED_DATASET_DIR = Path("fruit_analyzer/data/fruits_dataset_expanded")
OUTPUT_WEIGHTS_DIR = Path("fruit_analyzer/models/weights")
OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


class UnifiedFruitStockDataset(Dataset):
    """
    Unified dataset combining genuine in-the-wild stock photographs,
    background surface samples, and diverse Fruits-360 varieties with dynamic background augmentation.
    """

    def __init__(self, samples: List[Tuple[Path, int, bool]], transform=None, is_train: bool = True):
        self.samples = samples
        self.transform = transform
        self.is_train = is_train
        self.bg_augmenter = BackgroundAugmenter() if is_train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx, is_clean_bg = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback to dummy image if corrupted
            img = Image.new("RGB", (224, 224), (200, 200, 200))

        # Apply synthetic background blending only to clean-background laboratory crops
        if self.is_train and is_clean_bg and self.bg_augmenter is not None:
            if random.random() < 0.85:
                img = self.bg_augmenter.augment(img)

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.functional.to_tensor(img)

        return img_tensor, label_idx


def build_samples():
    train_samples = []
    val_samples = []

    # 1. Real Stock Dataset (in-the-wild stock photos + real background surfaces)
    if STOCK_DATASET_DIR.exists():
        for cls_name in CLASSES:
            cls_dir = STOCK_DATASET_DIR / cls_name
            if not cls_dir.exists():
                continue
            lbl = CLASS_TO_IDX[cls_name]
            files = sorted(list(cls_dir.glob("*.jpg")))
            random.seed(42)
            random.shuffle(files)
            # 80/20 train/val split for stock photos
            split_idx = int(len(files) * 0.8)
            for f in files[:split_idx]:
                train_samples.append((f, lbl, False))  # False = already has real background
            for f in files[split_idx:]:
                val_samples.append((f, lbl, False))

    # 2. Expanded Multi-Variety Dataset (Fruits-360 diverse varieties)
    exp_train_dir = EXPANDED_DATASET_DIR / "train"
    exp_val_dir = EXPANDED_DATASET_DIR / "val"

    if exp_train_dir.exists():
        for cls_name in CLASSES:
            if cls_name == "Background":
                continue
            cls_dir = exp_train_dir / cls_name
            if not cls_dir.exists():
                continue
            lbl = CLASS_TO_IDX[cls_name]
            for f in cls_dir.glob("*.jpg"):
                train_samples.append((f, lbl, True))  # True = clean background to augment

    if exp_val_dir.exists():
        for cls_name in CLASSES:
            if cls_name == "Background":
                continue
            cls_dir = exp_val_dir / cls_name
            if not cls_dir.exists():
                continue
            lbl = CLASS_TO_IDX[cls_name]
            for f in cls_dir.glob("*.jpg"):
                val_samples.append((f, lbl, True))

    return train_samples, val_samples


def main():
    print("=" * 70)
    print("FruitVision MobileNetV3-Large Real-Stock Training & Quantization")
    print("=" * 70)

    train_samples, val_samples = build_samples()
    print(f"Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples.")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")

    # Rich Photometric & Geometric Data Augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = UnifiedFruitStockDataset(train_samples, transform=train_transform, is_train=True)
    val_dataset = UnifiedFruitStockDataset(val_samples, transform=val_transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load MobileNetV3-Large with Squeeze-and-Excitation (SE) attention
    print("\nLoading MobileNetV3-Large pretrained on ImageNet-1k...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Replace classifier head for 16 classes (15 fruits + Background)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1024, len(CLASSES))
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0
    best_pth_path = OUTPUT_WEIGHTS_DIR / "best_mobilenetv3_stock.pth"

    # -------------------------------------------------------------
    # STAGE 1: Classifier Head Warmup (3 epochs, frozen backbone)
    # -------------------------------------------------------------
    print("\n[Stage 1/2] Warming up Classifier Head (3 epochs)...")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_head = optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=1e-2)

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            optimizer_head.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_head.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Warmup Ep [{epoch+1}/3] Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.1f}%")

    # -------------------------------------------------------------
    # STAGE 2: Progressive Fine-Tuning (Top SE blocks + Head, 3 epochs)
    # -------------------------------------------------------------
    print("\n[Stage 2/2] Fine-Tuning Top Feature Blocks with SE Attention (3 epochs)...")
    # Unfreeze top 4 bottleneck blocks (features[12:])
    for param in model.features[12:].parameters():
        param.requires_grad = True

    optimizer_ft = optim.AdamW([
        {"params": model.features[12:].parameters(), "lr": 6e-5},
        {"params": model.classifier.parameters(), "lr": 2e-4}
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=3, eta_min=1e-6)

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            optimizer_ft.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Finetune Ep [{epoch+1}/3] Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.1f}%")

        scheduler.step()

        # Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        train_acc = (correct / total) * 100.0
        print(f"Epoch [{epoch+1}/3] Finished - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_pth_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc:.2f}%)")

    print(f"\nTraining Complete! Best Real-Stock Validation Accuracy: {best_val_acc:.2f}%")

    # Save Model Config
    config_path = OUTPUT_WEIGHTS_DIR / "mobilenet_v3_config.json"
    id2label = {str(i): cls_name for i, cls_name in enumerate(CLASSES)}
    label2id = {cls_name: i for i, cls_name in enumerate(CLASSES)}
    config_dict = {
        "_name_or_path": "mobilenet_v3_large_real_stock",
        "architectures": ["MobileNetV3ForFruitClassification"],
        "num_classes": len(CLASSES),
        "id2label": id2label,
        "label2id": label2id,
        "classes": CLASSES,
        "dataset_name": "FruitVision Real-Stock + Fruits-360 Unified Dataset",
        "accuracy": f"{best_val_acc:.2f}%"
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")

    # Export to FP32 ONNX
    print("\nExporting to ONNX...")
    model.load_state_dict(torch.load(best_pth_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    fp32_onnx_path = OUTPUT_WEIGHTS_DIR / "mobilenet_v3_fruits_fp32.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    print(f"Exported FP32 ONNX ({fp32_onnx_path.stat().st_size / (1024*1024):.2f} MB).")

    # Quantize to dynamic INT8
    print("Quantizing to dynamic INT8...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_onnx_path = OUTPUT_WEIGHTS_DIR / "mobilenet_v3_quantized.onnx"

    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(quant_onnx_path),
        weight_type=QuantType.QInt8
    )
    print(f"Quantized ONNX saved ({quant_onnx_path.stat().st_size / (1024*1024):.2f} MB).")
    print("\nALL DONE! MobileNetV3-Large Real-Stock Model successfully trained and deployed!")


if __name__ == "__main__":
    main()
