#!/usr/bin/env python3
"""Train P1 BEV-conditioned K-head trajectory model (Hydra) with detailed TensorBoard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.data import BevTrajectoryNpzDataset
from lib.loss import p1_total_loss
from lib.model import BevTrajectoryNet
from lib.tb_utils import (
    _histogram_flat_float64,
    log_bev_channels,
    log_config_text,
    log_gradients_histograms,
    log_model_weights_histograms,
    log_trajectory_figure,
    total_grad_norm,
    val_pose_pointwise_l2,
)

# Relative paths in config (datasets/..., checkpoints/, logs/) are anchored here, not to CWD.
_PLANNER_BEV_ROOT = Path(__file__).resolve().parent


def _resolve_planner_path(p: str | Path) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_PLANNER_BEV_ROOT / path).resolve()


def _aux_or_none(aux_b: torch.Tensor, aux_dim: int) -> torch.Tensor | None:
    if aux_dim <= 0 or aux_b.shape[-1] == 0:
        return None
    return aux_b


def _tb_cfg(cfg: DictConfig) -> OmegaConf:
    d = cfg.train.get("tensorboard")
    if d is None:
        return OmegaConf.create(
            {
                "batch_scalars_interval": 1,
                "log_bev_every_epochs": 1,
                "log_traj_figure_every_epochs": 1,
                "log_weight_hist_every_epochs": 1,
                "log_grad_hist_every_epochs": 1,
                "log_grad_norm_every_batches": 1,
                "weight_histogram_substrings": ["conv", "fuse", "heads"],
                "bev_preview_max_side": 400,
                "log_graph": False,
            }
        )
    return OmegaConf.merge(
        OmegaConf.create(
            {
                "batch_scalars_interval": 1,
                "log_bev_every_epochs": 1,
                "log_traj_figure_every_epochs": 1,
                "log_weight_hist_every_epochs": 1,
                "log_grad_hist_every_epochs": 1,
                "log_grad_norm_every_batches": 1,
                "weight_histogram_substrings": ["conv", "fuse", "heads"],
                "bev_preview_max_side": 400,
                "log_graph": False,
            }
        ),
        d,
    )


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")
    print(
        f"[paths] train_dir={_resolve_planner_path(cfg.data.train_dir)}  "
        f"val_dir={_resolve_planner_path(cfg.data.val_dir)}  "
        f"save_dir={_resolve_planner_path(cfg.train.save_dir)}  "
        f"log_dir={_resolve_planner_path(cfg.train.log_dir)}"
    )

    tbc = _tb_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    aux_dim = int(cfg.model.aux_dim)
    train_dir = _resolve_planner_path(cfg.data.train_dir)
    val_dir = _resolve_planner_path(cfg.data.val_dir)
    train_ds = BevTrajectoryNpzDataset(train_dir, aux_dim=aux_dim)
    val_ds = BevTrajectoryNpzDataset(val_dir, aux_dim=aux_dim)

    if len(train_ds) == 0:
        raise RuntimeError(
            f"Empty train dataset at {train_dir} (config data.train_dir={cfg.data.train_dir!r}). Run: "
            "python3 prepare_data.py synthetic --out-train ... --out-val ..."
        )

    eff_bs = int(cfg.train.batch_size)
    drop_last = True
    if len(train_ds) < eff_bs:
        eff_bs = len(train_ds)
        drop_last = False
        print(f"[WARN] batch_size reduced to {eff_bs} (small dataset)")

    train_loader = DataLoader(
        train_ds,
        batch_size=eff_bs,
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eff_bs,
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    model = BevTrajectoryNet(
        h_bev=int(cfg.model.h_bev),
        w_bev=int(cfg.model.w_bev),
        c_in=int(cfg.model.c_in),
        aux_dim=aux_dim,
        num_heads=int(cfg.model.num_heads),
        horizon=int(cfg.model.horizon),
        stem_channels=tuple(int(x) for x in cfg.model.stem_channels),
        embed_dim=int(cfg.model.embed_dim),
        dropout=float(cfg.model.dropout),
    ).to(device)

    if cfg.train.pretrained_path:
        pre_path = _resolve_planner_path(str(cfg.train.pretrained_path))
        try:
            sd = torch.load(pre_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(pre_path, map_location=device)
        model.load_state_dict(sd)
        print(f"[INFO] Loaded {pre_path}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    save_dir = _resolve_planner_path(cfg.train.save_dir)
    log_dir = _resolve_planner_path(cfg.train.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir / ts)
    log_config_text(writer, OmegaConf.to_yaml(cfg), 0)
    try:
        htree = OmegaConf.to_container(cfg, resolve=True)
        writer.add_text("config/hparams_json", json.dumps(htree, indent=2, default=str), 0)
    except Exception:
        writer.add_text("config/hparams_json", "(unserializable)", 0)

    if bool(tbc.get("log_graph", False)):
        try:
            dummy = torch.zeros(
                1,
                int(cfg.model.c_in),
                int(cfg.model.h_bev),
                int(cfg.model.w_bev),
                device=device,
            )
            writer.add_graph(model, (dummy, None))
        except Exception as ex:
            print(f"[WARN] TensorBoard add_graph skipped: {ex}")

    print(f"TensorBoard: tensorboard --logdir {log_dir}   (this run: {log_dir / ts})")

    best_val = float("inf")
    patience = 0
    max_pat = int(cfg.train.early_stop_patience)
    best_path = save_dir / "best_model.pth"
    last_path = save_dir / "last_model.pth"

    huber_beta = float(cfg.train.loss.huber_beta)
    div_lambda = float(cfg.train.loss.div_lambda)
    div_d_min = float(cfg.train.loss.div_d_min)
    curv_lambda = float(cfg.train.loss.curv_lambda)
    aux_all_heads_lambda = float(cfg.train.loss.get("aux_all_heads_lambda", 0.0))

    batch_interval = max(1, int(tbc.batch_scalars_interval))
    grad_norm_interval = max(1, int(tbc.log_grad_norm_every_batches))
    wh = tbc.get("weight_histogram_substrings")
    if wh is None:
        w_hist_sub: Optional[List[str]] = ["conv", "fuse", "heads"]
    elif len(list(wh)) == 0:
        w_hist_sub = None
    else:
        w_hist_sub = list(wh)

    global_step = 0
    log_every = batch_interval
    grad_every = grad_norm_interval

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        train_tot = train_pose = train_aux = train_div = train_curv = 0.0
        n_tr = 0
        first_train_batch: tuple[torch.Tensor, ...] | None = None

        for batch_idx, (bev, traj, mode, aux) in enumerate(
            tqdm(train_loader, desc=f"Train {epoch+1}/{cfg.train.epochs}", leave=False)
        ):
            bev = bev.to(device)
            traj = traj.to(device)
            mode = mode.to(device)
            aux_b = _aux_or_none(aux.to(device), aux_dim)
            pred = model(bev, aux_b)
            loss, parts = p1_total_loss(
                pred,
                traj,
                mode,
                huber_beta,
                div_lambda,
                div_d_min,
                curv_lambda,
                None,
                aux_all_heads_lambda,
            )
            if first_train_batch is None:
                first_train_batch = (bev.detach(), traj.detach(), mode.detach(), pred.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            gnorm = total_grad_norm(model)
            if global_step % grad_every == 0:
                writer.add_scalar("train/batch/grad_norm_l2", gnorm, global_step)

            if (
                int(tbc.log_grad_hist_every_epochs) > 0
                and epoch % int(tbc.log_grad_hist_every_epochs) == 0
                and batch_idx == 0
            ):
                log_gradients_histograms(writer, model, global_step, w_hist_sub)

            optimizer.step()

            if global_step % log_every == 0:
                writer.add_scalar("train/batch/loss_total", parts["loss_total"], global_step)
                writer.add_scalar("train/batch/loss_pose", parts["loss_pose"], global_step)
                writer.add_scalar("train/batch/loss_div", parts["loss_div"], global_step)
                writer.add_scalar("train/batch/loss_curv", parts["loss_curv"], global_step)
                if aux_all_heads_lambda > 0.0:
                    writer.add_scalar("train/batch/loss_aux_all", parts["loss_aux_all"], global_step)
                writer.add_scalar("train/batch/lr", optimizer.param_groups[0]["lr"], global_step)

            train_tot += parts["loss_total"]
            train_pose += parts["loss_pose"]
            train_aux += parts["loss_aux_all"]
            train_div += parts["loss_div"]
            train_curv += parts["loss_curv"]
            n_tr += 1
            global_step += 1

        avg_tr = train_tot / max(n_tr, 1)
        avg_pose = train_pose / max(n_tr, 1)
        avg_aux = train_aux / max(n_tr, 1)
        avg_div = train_div / max(n_tr, 1)
        avg_curv = train_curv / max(n_tr, 1)

        writer.add_scalar("train/epoch/loss_total", avg_tr, epoch)
        writer.add_scalar("train/epoch/loss_pose", avg_pose, epoch)
        if aux_all_heads_lambda > 0.0:
            writer.add_scalar("train/epoch/loss_aux_all", avg_aux, epoch)
        writer.add_scalar("train/epoch/loss_div", avg_div, epoch)
        writer.add_scalar("train/epoch/loss_curv", avg_curv, epoch)

        if int(tbc.log_weight_hist_every_epochs) > 0 and epoch % int(tbc.log_weight_hist_every_epochs) == 0:
            log_model_weights_histograms(writer, model, epoch, w_hist_sub)

        # Train viz (first batch of epoch)
        if first_train_batch is not None and int(tbc.log_bev_every_epochs) > 0:
            if epoch % int(tbc.log_bev_every_epochs) == 0:
                bev0, traj0, mode0, pred0 = first_train_batch
                log_bev_channels(
                    writer,
                    bev0[0].cpu(),
                    epoch,
                    tag_prefix="train/viz/bev",
                    max_side=int(tbc.bev_preview_max_side),
                )
            if int(tbc.log_traj_figure_every_epochs) > 0 and epoch % int(tbc.log_traj_figure_every_epochs) == 0:
                _, traj0, mode0, pred0 = first_train_batch
                log_trajectory_figure(
                    writer,
                    traj0[0].cpu().numpy(),
                    pred0[0].cpu().numpy(),
                    int(mode0[0].item()),
                    epoch,
                    tag="train/viz/trajectory_xy",
                    title=f"train sample 0  epoch={epoch}",
                )

        model.eval()
        val_tot = val_pose = val_aux = val_div = val_curv = 0.0
        n_v = 0
        first_val: tuple[torch.Tensor, ...] | None = None
        val_err_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for bev, traj, mode, aux in tqdm(val_loader, desc="Val", leave=False):
                bev = bev.to(device)
                traj = traj.to(device)
                mode = mode.to(device)
                aux_b = _aux_or_none(aux.to(device), aux_dim)
                pred = model(bev, aux_b)
                _, parts = p1_total_loss(
                    pred,
                    traj,
                    mode,
                    huber_beta,
                    div_lambda,
                    div_d_min,
                    curv_lambda,
                    None,
                    aux_all_heads_lambda,
                )
                if first_val is None:
                    first_val = (bev.detach(), traj.detach(), mode.detach(), pred.detach())
                val_err_chunks.append(val_pose_pointwise_l2(pred, traj, mode).detach().cpu())
                val_tot += parts["loss_total"]
                val_pose += parts["loss_pose"]
                val_aux += parts["loss_aux_all"]
                val_div += parts["loss_div"]
                val_curv += parts["loss_curv"]
                n_v += 1

        avg_v = val_tot / max(n_v, 1) if n_v else float("inf")
        avg_vp = val_pose / max(n_v, 1) if n_v else float("inf")
        avg_va = val_aux / max(n_v, 1) if n_v else 0.0
        avg_vd = val_div / max(n_v, 1) if n_v else 0.0
        avg_vc = val_curv / max(n_v, 1) if n_v else 0.0

        writer.add_scalar("val/epoch/loss_total", avg_v, epoch)
        writer.add_scalar("val/epoch/loss_pose", avg_vp, epoch)
        if aux_all_heads_lambda > 0.0:
            writer.add_scalar("val/epoch/loss_aux_all", avg_va, epoch)
        writer.add_scalar("val/epoch/loss_div", avg_vd, epoch)
        writer.add_scalar("val/epoch/loss_curv", avg_vc, epoch)

        if val_err_chunks:
            all_err = torch.cat(val_err_chunks)
            writer.add_histogram(
                "val/hist/pointwise_l2_mean_per_sample",
                _histogram_flat_float64(all_err),
                epoch,
            )
            writer.add_scalar("val/epoch/mean_l2_teacher_head", float(all_err.mean()), epoch)
            writer.add_scalar("val/epoch/std_l2_teacher_head", float(all_err.std(unbiased=False)), epoch)

        if first_val is not None and int(tbc.log_bev_every_epochs) > 0:
            if epoch % int(tbc.log_bev_every_epochs) == 0:
                vb, _, _, _ = first_val
                log_bev_channels(
                    writer,
                    vb[0].cpu(),
                    epoch,
                    tag_prefix="val/viz/bev",
                    max_side=int(tbc.bev_preview_max_side),
                )
            if int(tbc.log_traj_figure_every_epochs) > 0 and epoch % int(tbc.log_traj_figure_every_epochs) == 0:
                _, vt, vm, vp = first_val
                log_trajectory_figure(
                    writer,
                    vt[0].cpu().numpy(),
                    vp[0].cpu().numpy(),
                    int(vm[0].item()),
                    epoch,
                    tag="val/viz/trajectory_xy",
                    title=f"val sample 0  epoch={epoch}",
                )

        if aux_all_heads_lambda > 0.0:
            print(
                f"Epoch {epoch+1:03d}: train_loss={avg_tr:.5f} train_pose={avg_pose:.5f} "
                f"train_aux={avg_aux:.5f} | val_loss={avg_v:.5f} val_pose={avg_vp:.5f} val_aux={avg_va:.5f}"
            )
        else:
            print(
                f"Epoch {epoch+1:03d}: train_loss={avg_tr:.5f} train_pose={avg_pose:.5f} | "
                f"val_loss={avg_v:.5f} val_pose={avg_vp:.5f}"
            )

        scheduler.step(avg_v)
        writer.add_scalar("train/epoch/lr", optimizer.param_groups[0]["lr"], epoch)
        if avg_v < float("inf") and avg_tr < float("inf"):
            writer.add_scalar("meta/val_train_loss_ratio", avg_v / max(avg_tr, 1e-8), epoch)

        if avg_v < best_val:
            best_val = avg_v
            patience = 0
            torch.save(model.state_dict(), best_path)
            print(f"  [save] best -> {best_path}")
        else:
            patience += 1
        torch.save(model.state_dict(), last_path)

        if max_pat > 0 and patience >= max_pat:
            print(f"[early-stop] no val improvement for {max_pat} epochs")
            break

    writer.add_text("run/summary", f"best_val_loss_total={best_val:.6f}\nfinal_global_step={global_step}", global_step)
    writer.close()
    print(f"TensorBoard log dir: {log_dir / ts}")
    print("Training finished.")


if __name__ == "__main__":
    main()
