"""
Clean, reusable training utilities shared by all compression stages
(teacher training, baseline student, distillation fine-tune, post-pruning fine-tune).
"""
import copy
import numpy as np
import torch
from torchvision import transforms

from src.utils import rmse, mae


def get_transforms(img_size=224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_tfm, val_tfm


def train_one_epoch(model, loader, optimizer, criterion, device):
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
    """Returns metrics dict, trues array, preds array."""
    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).squeeze(1).cpu().numpy()
            pred = np.clip(pred, 0, 116)
            preds_list.append(pred)
            trues_list.append(y.cpu().numpy())

    preds = np.concatenate(preds_list)
    trues = np.concatenate(trues_list)

    return {"rmse": rmse(trues, preds), "mae": mae(trues, preds)}, trues, preds


def train_two_stage(model, train_loader, val_loader, cfg, device, verbose=True):
    """
    Two-stage training:
      Stage 1 – train head only (backbone frozen).
      Stage 2 – fine-tune all layers starting from best stage-1 weights.

    cfg keys used:
      training.epochs_head, training.epochs_finetune,
      training.lr_head, training.lr_finetune,
      training.weight_decay, training.patience,
      training.beta_smooth_l1
    """
    tcfg = cfg["training"]
    criterion = torch.nn.SmoothL1Loss(beta=tcfg["beta_smooth_l1"])

    def _run_stage(epochs, lr, unfreeze_all, stage_name):
        if epochs == 0:
            return float("inf")
        if unfreeze_all:
            for p in model.parameters():
                p.requires_grad = True
            opt_params = model.parameters()
        else:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
            opt_params = model.fc.parameters()

        optimizer = torch.optim.AdamW(opt_params, lr=lr,
                                      weight_decay=tcfg["weight_decay"])

        best_rmse = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics, _, _ = evaluate(model, val_loader, device)

            if verbose:
                print(f"[{stage_name}] epoch {epoch}/{epochs}  "
                      f"loss={loss:.4f}  val_rmse={metrics['rmse']:.4f}  "
                      f"val_mae={metrics['mae']:.4f}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= tcfg["patience"]:
                if verbose:
                    print(f"[{stage_name}] early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        return best_rmse

    if verbose:
        print("=== Stage 1: head only ===")
    best_rmse_head = _run_stage(
        tcfg["epochs_head"], tcfg["lr_head"],
        unfreeze_all=False, stage_name="head"
    )

    if verbose:
        print(f"\n=== Stage 2: full fine-tune (starting from head RMSE {best_rmse_head:.4f}) ===")
    best_rmse_ft = _run_stage(
        tcfg["epochs_finetune"], tcfg["lr_finetune"],
        unfreeze_all=True, stage_name="finetune"
    )

    if verbose:
        print(f"\nDone. Best val RMSE: {best_rmse_ft:.4f}")

    return model
