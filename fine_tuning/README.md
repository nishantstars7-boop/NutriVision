# 🍎 MobileNetV2 Fruit Fine-Tuning & Export Guide

This directory contains complete scripts and a Google Colab notebook to fine-tune **MobileNetV2** on the **Fruits-360** dataset combined with your custom fruit photos, and export the trained model to **ONNX** (with INT8 quantization) and **TFLite**.

---

## 📁 Files in This Directory

| File | Purpose |
| :--- | :--- |
| `Fruit_Classifier_MobileNetV2_Colab.ipynb` | **Interactive Google Colab Notebook** (Free GPU 1-click training, curves, export, download). |
| `train_mobilenet_fruits360.py` | Standalone Python training script with CLI arguments. |
| `README.md` | Step-by-step instructions and best practices. |

---

## 🚀 Option 1: Run on Google Colab (Recommended)

Google Colab provides a free NVIDIA T4 GPU that can train the model in ~8-12 minutes:

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `Fruit_Classifier_MobileNetV2_Colab.ipynb`.
3. In Colab, go to **Runtime** > **Change runtime type** > Select **T4 GPU**.
4. Run all cells sequentially (`Ctrl + F9`):
   - Cell 1-2: Installs dependencies and checks GPU.
   - Cell 3: Downloads Fruits-360 dataset automatically.
   - Cell 4-5: Applies heavy data augmentations (rotation, lighting, blur, affine).
   - Cell 6: Runs two-stage transfer learning (warmup head + fine-tuning backbone).
   - Cell 7: Exports to ONNX and quantizes to INT8 (~3.5 MB).
   - Cell 8: Exports to TFLite (for mobile edge devices).
   - Cell 9: Downloads `mobilenet_v2_fruits_quantized.onnx` and `fruit_classes.json` to your computer.

### Adding Your Own Fruit Photos in Colab
To add photos of your own fruits:
```python
# Create a folder for your custom fruit
!mkdir -p custom_photos/Apple
!mkdir -p custom_photos/Banana

# Upload your photos, then merge them into the training set:
!cp -r custom_photos/* /root/.cache/kagglehub/datasets/moltean/fruits/versions/latest/fruits-360_dataset/fruits-360/Training/
```

---

## 💻 Option 2: Run Locally via Python CLI

If you have a local GPU with CUDA:

```bash
# 1. Install PyTorch & ONNX tools
pip install torch torchvision onnx onnxruntime albumentations

# 2. Run training script
python fruit_analyzer/fine_tuning/train_mobilenet_fruits360.py \
    --data-dir ./fruits-360 \
    --batch-size 64 \
    --epochs 10 \
    --output-onnx mobilenet_v2_fruits.onnx
```

---

## 🔄 Deploying the Exported Model to Fruit Analyzer App

Once exported:
1. Copy `mobilenet_v2_fruits_quantized.onnx` to `fruit_analyzer/models/weights/mobilenet_v2_quantized.onnx`.
2. Copy `fruit_classes.json` to `fruit_analyzer/models/weights/mobilenet_v2_config.json`.
3. Launch the Streamlit app — it will automatically detect and load your newly fine-tuned weights!
