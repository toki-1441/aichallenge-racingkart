"""Train VelocityFormer (BERT-tiny) for trajectory-conditioned control regression.

Mirrors the tiny_lidar_net training loop: Hydra config, TensorBoard logging,
early stopping, best/last checkpoint saving.
"""

from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.data import MultiSeqConcatDataset
from lib.loss import HuberLoss
from lib.model import VelocityFormer


def clean_numerical_tensor(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf values to keep training stable."""
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    flip_prob = cfg.augment.flip_prob if cfg.augment.enable else 0.0

    train_dataset = MultiSeqConcatDataset(
        cfg.data.train_dir,
        label_type=cfg.model.label_type,
        max_steering=cfg.trajectory.max_steering,
        min_steering=cfg.trajectory.min_steering,
        flip_prob=flip_prob,
    )
    val_dataset = MultiSeqConcatDataset(
        cfg.data.val_dir,
        label_type=cfg.model.label_type,
        max_steering=cfg.trajectory.max_steering,
        min_steering=cfg.trajectory.min_steering,
        flip_prob=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = VelocityFormer(
        pretrained_model=cfg.model.pretrained_model,
        input_size=cfg.model.input_size,
        num_labels=cfg.model.num_labels,
        load_pretrained_weights=True,
    ).to(device)

    if cfg.train.pretrained_path:
        model.load_state_dict(torch.load(cfg.train.pretrained_path, map_location=device))
        print(f"[INFO] Loaded pretrained model from {cfg.train.pretrained_path}")

    criterion = HuberLoss(beta=cfg.train.loss.huber_beta)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg.train.save_dir).expanduser().resolve()
    log_dir = Path(cfg.train.log_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    suffix = cfg.model.label_type
    best_path = save_dir / f"best_model_{suffix}.pth"
    last_path = save_dir / f"last_model_{suffix}.pth"

    with SummaryWriter(log_dir / f"{suffix}_{timestamp}") as writer:
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = cfg.train.get("early_stop_patience", 10)

        for epoch in range(cfg.train.epochs):
            model.train()
            train_loss = 0.0

            for input_ids, targets in tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}/{cfg.train.epochs}"):
                input_ids = input_ids.to(device, dtype=torch.long)
                targets = targets.to(device)

                input_ids = clean_numerical_tensor(input_ids)
                targets = clean_numerical_tensor(targets)

                outputs = model(input_ids)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / max(1, len(train_loader))
            avg_val_loss = validate(model, val_loader, device, criterion)

            print(f"Epoch {epoch + 1:03d}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
            writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
            writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)
                print(f"[SAVE] Best model updated: {best_path} (val_loss={best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            torch.save(model.state_dict(), last_path)
            if patience_counter >= max_patience:
                print(f"[EarlyStop] No improvement for {max_patience} epochs.")
                break

    print("Training finished.")


def validate(model, loader, device, criterion) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, targets in tqdm(loader, desc="[Val]", leave=False):
            input_ids = input_ids.to(device, dtype=torch.long)
            targets = targets.to(device)
            input_ids = clean_numerical_tensor(input_ids)
            targets = clean_numerical_tensor(targets)
            outputs = model(input_ids)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


if __name__ == "__main__":
    main()
