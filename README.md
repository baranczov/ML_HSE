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

## Project Structure

```
ML_HSE/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies with pinned versions
├── .gitignore                # Git ignore rules
├── configs/
│   └── default.yaml          # Experiment configuration (hyperparameters)
├── src/
│   ├── __init__.py
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Model evaluation script
│   ├── utils.py              # Helper functions (metrics, visualization)
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset for UTKFace
│   │   └── preprocessing.py  # Label parsing, data splitting
│   └── models/
│       └── resnet_model.py   # Model architecture (ResNet18)
├── notebooks/                # Jupyter notebooks for EDA
│   ├── notebook_v1.ipynb
│   └── notebook_v2.ipynb
├── examples/                 # Reference solutions
│   ├── example_1.ipynb
│   └── example_2.ipynb
├── logs/                     # Experiment logs (TensorBoard)
├── models/                   # Saved model weights
└── results/                  # Evaluation results (predictions, metrics)
```

## Exploratory Data Analysis

### Age Distribution
The dataset shows two distinct peaks:
- **Children (0-12 years)** — first large cluster
- **Young adults (20-45 years)** — second peak
- Long tail toward elderly ages (80+ years)

![Age distribution](https://via.placeholder.com/600x300?text=Age+Distribution+Plot)
*Placeholder: insert age distribution histogram*

### Gender Distribution
| Gender | Count | Percentage |
|--------|-------|------------|
| Male (0) | 12,389 | 55.3% |
| Female (1) | 11,314 | 44.7% |

### Race Distribution
| Race | Count | Percentage |
|------|-------|------------|
| White (0) | 10,077 | 45.3% |
| Black (1) | 4,525 | 20.3% |
| Asian (3) | 3,975 | 17.9% |
| Indian (2) | 3,434 | 15.4% |
| Other (4) | 1,692 | 7.6% |

### Age by Gender Distribution
![Age by gender](https://via.placeholder.com/600x300?text=Age+by+Gender+Plot)
*Placeholder: insert stacked bar chart of age bins by gender*

## Error Analysis

### Key Findings
1. **Children (0-12 years)**: RMSE = 5.7-6.1 — facial features change rapidly, making accurate age estimation difficult
2. **Elderly (80+ years)**: RMSE = 13.3 — extremely low sample count (134 images) leads to poor generalization
3. **Age group boundaries**: Most confusion occurs between adjacent bins (e.g., 19 and 20, 29 and 30)

### Confusion Matrix (Age Bins)
*Placeholder: insert confusion matrix from your notebook*

### Recommendations for Improvement
- **Class imbalance**: Use weighted loss or oversampling for rare age groups
- **Data augmentation**: More aggressive augmentation for underrepresented ages
- **Ordinal regression**: Treat as ordinal classification instead of pure regression
- **Uncertainty estimation**: Predict age distribution rather than single value

## Technical Details

### Model Architecture
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Head**: Dropout(0.2) + Linear(512 → 1)

### Training Strategy
1. **Stage 1 (Head only)**: 3 epochs, LR = 1e-3
2. **Stage 2 (Fine-tune)**: 7 epochs, LR = 3e-4
- **Optimizer**: AdamW
- **Weight decay**: 1e-4
- **Loss**: Smooth L1 Loss (beta=5.0)

### Data Augmentation
```python
transforms.Compose([
    Resize(256),
    RandomResizedCrop(224, scale=(0.85, 1.0)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

## Experiment Tracking

### Logged Artifacts
- Training loss per epoch
- Validation RMSE and MAE
- Best model weights (by RMSE)
- Predictions on validation set
- Configuration snapshots

### Example Logs (TensorBoard)
*Placeholder: insert TensorBoard screenshot*

## References

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet18 Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

## Authors

- **Баранцов Данил**
- **Костырин Николай**

Group 08, ML Course, HSE
