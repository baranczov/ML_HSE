import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet_model import create_resnet18, get_device
from src.data.dataset import AgeDataset
from src.data.preprocessing import load_data, create_age_bins, train_val_split
from src.utils import rmse, mae, acc_at_k, compute_metrics_by_bin, plot_confmat

def get_transform(img_size=224):
    """Get validation transform"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    return transform

def evaluate_model(config, model_path):
    """Evaluate trained model on validation set"""
    device = get_device()
    
    # Load data
    df, _ = load_data(config["paths"]["data_dir"])
    df, labels = create_age_bins(df)
    
    # Train/val split
    _, val_df = train_val_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["seed"]
    )
    
    # Create dataloader
    transform = get_transform(config["data"]["img_size"])
    val_ds = AgeDataset(val_df, transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    # Load model
    model = create_resnet18(
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Predict
    preds = []
    trues = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x).squeeze(1).cpu().numpy()
            pred = np.clip(pred, 0, 116)
            preds.append(pred)
            trues.append(y.numpy())
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # Compute metrics
    metrics = {
        "rmse": rmse(trues, preds),
        "mae": mae(trues, preds),
        "acc@5": acc_at_k(trues, preds, k=5),
        "acc@10": acc_at_k(trues, preds, k=10),
    }
    
    # Metrics by age bin
    val_df = val_df.reset_index(drop=True)
    val_df["pred_age"] = preds
    
    by_bin = compute_metrics_by_bin(val_df, "age", "pred_age", 
                                     bins=[0,3,13,20,30,40,50,60,70,80,117],
                                     labels=labels)
    
    return metrics, by_bin, val_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    metrics, by_bin, val_df = evaluate_model(config, args.model)
    
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== RMSE by Age Bin ===")
    print(by_bin[["n", "rmse"]])

if __name__ == "__main__":
    main()
