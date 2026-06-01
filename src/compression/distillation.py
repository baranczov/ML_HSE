"""
Knowledge Distillation: ResNet50 (teacher) → ResNet18 (student).

Two KD-loss variants:
  "response" — MSE between student and teacher age predictions (soft targets).
  "feature"  — cosine / MSE between backbone embeddings, with optional linear
               projection when teacher/student embedding dims differ (2048 vs 512).
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import rmse, mae
from src.compression.training import train_one_epoch, evaluate


class _EmbeddingExtractor(nn.Module):
    """Wraps a ResNet to return penultimate features (before fc)."""

    def __init__(self, model):
        super().__init__()
        # Everything except fc
        self.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool,
        )
        self.fc = model.fc

    def forward_embedding(self, x):
        feat = self.backbone(x)
        return torch.flatten(feat, 1)

    def forward(self, x):
        feat = self.forward_embedding(x)
        return self.fc(feat)


class DistillationLoss(nn.Module):
    """
    L_total = lambda_task * SmoothL1(student_pred, y)
            + lambda_kd   * L_KD(student, teacher)

    kd_type="response": L_KD = MSE(student_pred, teacher_pred.detach())
    kd_type="feature":  L_KD = 1 - cosine_similarity(student_emb, teacher_emb_proj)
                        (with optional linear projection if dims differ)
    """

    def __init__(
        self,
        task_loss,
        lambda_task=1.0,
        lambda_kd=1.0,
        kd_type="response",
        student_emb_dim=None,
        teacher_emb_dim=None,
    ):
        super().__init__()
        self.task_loss = task_loss
        self.lambda_task = lambda_task
        self.lambda_kd = lambda_kd
        self.kd_type = kd_type

        self.projection = None
        if kd_type == "feature" and student_emb_dim and teacher_emb_dim:
            if student_emb_dim != teacher_emb_dim:
                # Project teacher embedding to student space before comparing
                self.projection = nn.Linear(teacher_emb_dim, student_emb_dim, bias=False)

    def forward(self, student_pred, teacher_pred, y,
                student_emb=None, teacher_emb=None):
        l_task = self.task_loss(student_pred, y)

        if self.kd_type == "response":
            l_kd = F.mse_loss(student_pred, teacher_pred.detach())
        elif self.kd_type == "feature":
            if student_emb is None or teacher_emb is None:
                raise ValueError("kd_type='feature' requires student_emb and teacher_emb")
            t_emb = teacher_emb.detach()
            if self.projection is not None:
                t_emb = self.projection(t_emb)
            # Normalize
            s_norm = F.normalize(student_emb, dim=1)
            t_norm = F.normalize(t_emb, dim=1)
            l_kd = 1.0 - (s_norm * t_norm).sum(dim=1).mean()
        else:
            raise ValueError(f"Unknown kd_type: {self.kd_type}")

        return self.lambda_task * l_task + self.lambda_kd * l_kd, l_task, l_kd


def _make_extractors(student, teacher):
    s_ext = _EmbeddingExtractor(student)
    t_ext = _EmbeddingExtractor(teacher)
    return s_ext, t_ext


def train_distillation(student, teacher, train_loader, val_loader, cfg, device,
                       verbose=True):
    """
    Train student with knowledge from frozen teacher.

    cfg keys:
      distillation.kd_type, distillation.lambda_task, distillation.lambda_kd
      training.epochs_finetune, training.lr_finetune, training.weight_decay,
      training.beta_smooth_l1, training.patience
    """
    dcfg = cfg["distillation"]
    tcfg = cfg["training"]

    kd_type = dcfg.get("kd_type", "response")

    # Teacher is frozen
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = student.to(device)

    # Unfreeze all student layers for distillation fine-tuning
    for p in student.parameters():
        p.requires_grad = True

    task_loss_fn = nn.SmoothL1Loss(beta=tcfg["beta_smooth_l1"])

    # Get embedding dims for feature KD
    from src.models.resent_model import get_embedding_dim
    s_dim = get_embedding_dim(student)
    t_dim = get_embedding_dim(teacher)

    kd_loss_fn = DistillationLoss(
        task_loss=task_loss_fn,
        lambda_task=float(dcfg.get("lambda_task", 1.0)),
        lambda_kd=float(dcfg.get("lambda_kd", 1.0)),
        kd_type=kd_type,
        student_emb_dim=s_dim,
        teacher_emb_dim=t_dim,
    ).to(device)

    # Build embedding extractors (reuse backbone references)
    s_ext, t_ext = _make_extractors(student, teacher)
    s_ext = s_ext.to(device)
    t_ext = t_ext.to(device)
    t_ext.eval()

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(kd_loss_fn.parameters()),
        lr=float(tcfg["lr_finetune"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    best_rmse = float("inf")
    best_state = copy.deepcopy(student.state_dict())
    bad_epochs = 0
    epochs = tcfg["epochs_finetune"]
    patience = tcfg["patience"]

    for epoch in range(1, epochs + 1):
        student.train()
        s_ext.train()

        total_loss = total_task = total_kd = 0.0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                t_pred = teacher(x).squeeze(1)

            if kd_type == "feature":
                with torch.no_grad():
                    t_emb = t_ext.forward_embedding(x)
                # Compute embedding once; reuse for both head pass and KD loss
                # to avoid running the backbone twice (which doubles BN stat updates)
                s_emb = s_ext.forward_embedding(x)
                s_pred = student.fc(s_emb).squeeze(1)
            else:
                s_pred = student(x).squeeze(1)
                s_emb = t_emb = None

            loss, l_task, l_kd = kd_loss_fn(s_pred, t_pred, y, s_emb, t_emb)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_task += l_task.item() * bs
            total_kd   += l_kd.item() * bs
            total_n    += bs

        avg_loss = total_loss / total_n
        avg_task = total_task / total_n
        avg_kd   = total_kd   / total_n

        metrics, _, _ = evaluate(student, val_loader, device)

        if verbose:
            print(f"[distill] epoch {epoch}/{epochs}  "
                  f"loss={avg_loss:.4f}  task={avg_task:.4f}  kd={avg_kd:.4f}  "
                  f"val_rmse={metrics['rmse']:.4f}  val_mae={metrics['mae']:.4f}")

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = copy.deepcopy(student.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            if verbose:
                print(f"[distill] early stopping at epoch {epoch}")
            break

    student.load_state_dict(best_state)
    if verbose:
        print(f"Distillation done. Best val RMSE: {best_rmse:.4f}")

    return student
