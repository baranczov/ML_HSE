# ML_HSE: Age Prediction from Face Images

Computer Vision project for predicting age from facial images using transfer learning.

![Face age estimation](https://github.com/user-attachments/assets/f8c3c5ed-7ed3-414e-aa90-00a4af27bf38)

## Project Overview

| | |
|---|---|
| **Task** | Regression age prediction (in years) |
| **Dataset** | [UTKFace](https://susanqq.github.io/UTKFace/) - 23,708 face images with age, gender, and race labels |
| **Approach** | Transfer learning with ResNet18 pretrained on ImageNet |
| **Loss** | Smooth L1 Loss (Huber loss) for robustness to outliers |
| **Framework** | PyTorch + torchvision |

## Results

| Metric | Value |
|--------|-------|
| RMSE | 7.38 |
| MAE | 5.20 |

### Performance by Age Group

| Age Bin | Count | RMSE |
|---------|-------|------|
| 0-2 | 321 | 3.95 |
| 3-12 | 362 | 5.71 |
| 13-19 | 236 | 6.12 |
| 20-29 | 1469 | 5.51 |
| 30-39 | 907 | 6.92 |
| 40-49 | 449 | 8.13 |
| 50-59 | 460 | 9.80 |
| 60-69 | 263 | 10.75 |
| 70-79 | 140 | 10.87 |
| 80+ | 134 | 13.30 |

**Observation**: Model performs best on ages 20-50 (most frequent in dataset) and struggles with children (0-12) and elderly (80+), likely due to class imbalance and rapid facial changes.

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ML_HSE.git
cd ML_HSE
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Download UTKFace from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Place the images in `data/UTKFace/` or update config path

### 4. Train the model
```bash
python src/train.py --config configs/default.yaml
```

### 5. Evaluate a trained model
```bash
python src/evaluate.py --config configs/default.yaml --model models/best_model.pth
```

## References

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet18 Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

## Authors

- **Баранцов Данил**
- **Костырин Николай**

Group 08, ML Course, HSE
