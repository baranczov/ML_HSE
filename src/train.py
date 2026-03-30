import os
import copy
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet_model import create_resnet18, get_device
from src.data.dataset import AgeDataset
from src.data.preprocessing import load_data, create_age_bins, train_val_split
from src.utils import rmse, mae

def get_transforms(img_size=224):
    """Get train and validation transforms"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    return train_transform, val_transform

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_n = 0
    
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred = model(x).squeeze(1)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    
    return total_loss / total_n

def evaluate(model, loader, device):
    """Evaluate model on validation set"""
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).squeeze(1).cpu().numpy()
            pred = np.clip(pred, 0, 116)
            preds.append(pred)
            trues.append(y.numpy())
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    return {
        "rmse": rmse(trues, preds),
        "mae": mae(trues, preds)
    }, trues, preds

def train_model(config):
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(config["data"]["seed"])
    np.random.seed(config["data"]["seed"])
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df, bad = load_data(config["paths"]["data_dir"])
    print(f"Loaded {len(df)} images ({bad} failed to parse)")
    
    # Add age bins
    df, labels = create_age_bins(df)
    
    # Train/val split
    train_df, val_df = train_val_split(
        df, 
        test_size=config["data"]["test_size"],
        random_state=config["data"]["seed"]
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Get transforms
    train_tfms, val_tfms = get_transforms(config["data"]["img_size"])
    
    # Create datasets
    train_ds = AgeDataset(train_df, train_tfms)
    val_ds = AgeDataset(val_df, val_tfms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    # Create model
    model = create_resnet18(
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"]
    )
    model = model.to(device)
    
    criterion = nn.SmoothL1Loss(beta=config["training"]["beta_smooth_l1"])
    
    # Stage 1: Train only head
    print("\n=== Stage 1: Training head only ===")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=config["training"]["lr_head"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    best_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    
    for epoch in range(1, config["training"]["epochs_head"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics, _, _ = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}: loss={loss:.4f}, val_rmse={metrics['rmse']:.4f}, val_mae={metrics['mae']:.4f}")
        
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        if bad_epochs >= config["training"]["patience"]:
            print("Early stopping")
            break
    
    # Stage 2: Fine-tune all layers
    print("\n=== Stage 2: Fine-tuning all layers ===")
    model.load_state_dict(best_state)
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr_finetune"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    best_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    
    for epoch in range(1, config["training"]["epochs_finetune"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics, trues, preds = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}: loss={loss:.4f}, val_rmse={metrics['rmse']:.4f}, val_mae={metrics['mae']:.4f}")
        
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        if bad_epochs >= config["training"]["patience"]:
            print("Early stopping")
            break
    
    # Save best model
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    torch.save(best_state, os.path.join(config["paths"]["models_dir"], "best_model.pth"))
    print(f"\nModel saved to {config['paths']['models_dir']}/best_model.pth")
    
    # Save results
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    np.save(os.path.join(config["paths"]["results_dir"], "val_predictions.npy"), preds)
    np.save(os.path.join(config["paths"]["results_dir"], "val_targets.npy"), trues)
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    model, metrics = train_model(config)
    print(f"\nFinal results: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

if __name__ == "__main__":
    main()
