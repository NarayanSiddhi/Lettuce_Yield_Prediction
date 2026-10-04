#!/usr/bin/env python3
"""
Aim 2 Phase B (v3) — FOV-aware Δ-mask residual forecast.

Predict soft ExGR growth residual r = soft_{t+4} - soft_t, then
  soft_hat = clip(soft_t + r, 0, 1)
instead of absolute future mask. Soft maps are percentile-normalized ExGR
(FOV / illumination more stable than hard masks).

Take: residual growth (common in crop trajectory models).
Skip: RGB Pix2Pix / full diffusion.
Improve: beat persist Dice & veg-MAE via residual U-Net (+ optional TinyT).

Usage:
  python3 ml/forecast_images/train_aim2_phase_b_delta_mask.py --trial trial2 --model unet --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[2]
IMG_DIR = Path(__file__).resolve().parent
OUT_ROOT = IMG_DIR / "aim2_phase_b_delta"


def rgb_to_soft_exgr(img_chw01: np.ndarray) -> np.ndarray:
    """CHW [0,1] → soft ExGR map in [0,1] (percentile-normalized)."""
    r, g, b = img_chw01[0], img_chw01[1], img_chw01[2]
    exgr = 2.0 * g - r - b
    lo, hi = np.percentile(exgr, [20, 95])
    if hi - lo < 1e-6:
        return np.zeros_like(exgr, dtype=np.float32)
    soft = (exgr - lo) / (hi - lo)
    return np.clip(soft, 0, 1).astype(np.float32)


def soft_to_hard(soft: np.ndarray | torch.Tensor, thr: float = 0.45):
    if isinstance(soft, torch.Tensor):
        return (soft > thr).float()
    return (soft > thr).astype(np.float32)


class DeltaMaskDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, climate_cols: list[str], norm: dict, size: int):
        self.df = manifest.reset_index(drop=True)
        self.climate_cols = climate_cols
        self.norm = norm
        self.size = size

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, rel: str) -> np.ndarray:
        img = Image.open(PROJECT / rel).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr.transpose(2, 0, 1)

    def _climate(self, row) -> torch.Tensor:
        vals = []
        for c in self.climate_cols:
            v = float(row[c])
            m = self.norm[c]["mean"]
            s = self.norm[c]["std"] or 1.0
            vals.append((v - m) / s)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x01 = self._load_rgb(row["input_png"])
        y01 = self._load_rgb(row["target_png"])
        soft_t = rgb_to_soft_exgr(x01)
        soft_f = rgb_to_soft_exgr(y01)
        # FOV scale: normalize by current canopy mass so growth is relative
        mass = float(soft_t.mean()) + 1e-3
        soft_t_n = soft_t / mass
        soft_f_n = soft_f / mass
        # residual in roughly [-1,1] after norm; we train on unnormalized residual too
        residual = soft_f - soft_t  # absolute soft residual (primary)
        residual_n = soft_f_n - soft_t_n
        return {
            "input": torch.from_numpy(x01 * 2.0 - 1.0),
            "soft_t": torch.from_numpy(soft_t).unsqueeze(0),
            "soft_f": torch.from_numpy(soft_f).unsqueeze(0),
            "residual": torch.from_numpy(residual).unsqueeze(0),
            "residual_n": torch.from_numpy(residual_n).unsqueeze(0),
            "mass": torch.tensor([mass], dtype=torch.float32),
            "climate": self._climate(row),
            "stem": Path(row["input_png"]).stem,
        }


def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualUNet(nn.Module):
    """Predict soft residual Δ; reconstruct soft_f = clip(soft_t + Δ, 0, 1)."""

    def __init__(self, n_climate: int = 12, base: int = 32):
        super().__init__()
        self.n_climate = n_climate
        in_ch = 3 + 1 + n_climate
        b = base
        self.enc1 = nn.Sequential(conv_bn_relu(in_ch, b), conv_bn_relu(b, b))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(conv_bn_relu(b, b * 2), conv_bn_relu(b * 2, b * 2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(conv_bn_relu(b * 2, b * 4), conv_bn_relu(b * 4, b * 4))
        self.pool3 = nn.MaxPool2d(2)
        self.bot = nn.Sequential(conv_bn_relu(b * 4, b * 8), conv_bn_relu(b * 8, b * 8))
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.dec3 = nn.Sequential(conv_bn_relu(b * 8, b * 4), conv_bn_relu(b * 4, b * 4))
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.dec2 = nn.Sequential(conv_bn_relu(b * 4, b * 2), conv_bn_relu(b * 2, b * 2))
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.dec1 = nn.Sequential(conv_bn_relu(b * 2, b), conv_bn_relu(b, b))
        # tanh residual head
        self.head = nn.Sequential(nn.Conv2d(b, 1, 1), nn.Tanh())

    def forward(self, x, soft_t, climate):
        bsz, _, h, w = x.shape
        c = climate.view(bsz, -1, 1, 1).expand(-1, -1, h, w)
        z = torch.cat([x, soft_t, c], dim=1)
        e1 = self.enc1(z)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bot = self.bot(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(bot), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        # scale tanh to allow larger growth/shrink
        delta = self.head(d1) * 0.75
        soft_hat = torch.clamp(soft_t + delta, 0.0, 1.0)
        return delta, soft_hat


class ResidualTinyT(nn.Module):
    def __init__(self, n_climate: int = 12, img_size: int = 128, patch: int = 8, dim: int = 128, depth: int = 4, heads: int = 4):
        super().__init__()
        self.patch = patch
        self.n_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(4, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.clim_fc = nn.Linear(n_climate, dim)
        side = img_size // patch
        self.side = side
        self.dec = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, soft_t, climate):
        z = torch.cat([x, soft_t], dim=1)
        tok = self.proj(z)
        b, d, s, _ = tok.shape
        tok = tok.flatten(2).transpose(1, 2) + self.pos
        tok = tok + self.clim_fc(climate).unsqueeze(1)
        tok = self.encoder(tok)
        feat = tok.transpose(1, 2).reshape(b, d, s, s)
        delta = self.dec(feat) * 0.75
        soft_hat = torch.clamp(soft_t + delta, 0.0, 1.0)
        return delta, soft_hat


def dice_from_soft(soft_hat, soft_f, thr=0.45):
    pred = (soft_hat > thr).float()
    tgt = (soft_f > thr).float()
    inter = (pred * tgt).sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + 1e-6)
    union = pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return dice, iou, pred, tgt


def batch_metrics(soft_hat, soft_t, soft_f):
    dice, iou, pred, tgt = dice_from_soft(soft_hat, soft_f)
    persist = soft_t
    dice_p, iou_p, _, _ = dice_from_soft(persist, soft_f)
    veg_t = soft_f.mean(dim=(1, 2, 3))
    veg_h = soft_hat.mean(dim=(1, 2, 3))
    veg_p = soft_t.mean(dim=(1, 2, 3))
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "dice_persist": float(dice_p.mean().item()),
        "iou_persist": float(iou_p.mean().item()),
        "veg_mae": float((veg_h - veg_t).abs().mean().item()),
        "veg_mae_persist": float((veg_p - veg_t).abs().mean().item()),
        "soft_l1": float((soft_hat - soft_f).abs().mean().item()),
        "soft_l1_persist": float((soft_t - soft_f).abs().mean().item()),
    }


def build_model(name: str, n_climate: int, size: int) -> nn.Module:
    if name == "unet":
        return ResidualUNet(n_climate=n_climate)
    if name == "tinyt":
        return ResidualTinyT(n_climate=n_climate, img_size=size)
    raise SystemExit(f"unknown model {name}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    keys = [
        "dice",
        "iou",
        "dice_persist",
        "iou_persist",
        "veg_mae",
        "veg_mae_persist",
        "soft_l1",
        "soft_l1_persist",
        "loss",
    ]
    agg = {k: 0.0 for k in keys}
    n = 0
    for batch in loader:
        x = batch["input"].to(device)
        st = batch["soft_t"].to(device)
        sf = batch["soft_f"].to(device)
        res = batch["residual"].to(device)
        c = batch["climate"].to(device)
        delta, soft_hat = model(x, st, c)
        loss = F.l1_loss(delta, res) + F.l1_loss(soft_hat, sf) + (1.0 - dice_from_soft(soft_hat, sf)[0].mean())
        met = batch_metrics(soft_hat, st, sf)
        bs = x.size(0)
        agg["loss"] += float(loss.item()) * bs
        for k, v in met.items():
            agg[k] += v * bs
        n += bs
    return {k: v / max(n, 1) for k, v in agg.items()}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    trial = args.trial
    size = args.size
    ds_root = IMG_DIR / "dataset" / f"{trial}_128"
    blob = json.loads((ds_root / "climate_norm.json").read_text())
    climate_norm = blob["climate"]
    manifest = pd.read_csv(ds_root / "manifest.csv")
    climate_cols = sorted([c for c in climate_norm.keys() if c in manifest.columns])
    n_clim = len(climate_cols)

    def split_df(name):
        return manifest[manifest["split"] == name].reset_index(drop=True)

    train_ds = DeltaMaskDataset(split_df("train"), climate_cols, climate_norm, size)
    val_ds = DeltaMaskDataset(split_df("val"), climate_cols, climate_norm, size)
    test_ds = DeltaMaskDataset(split_df("test"), climate_cols, climate_norm, size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model, n_clim, size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_dir = OUT_ROOT / f"{trial}_{args.model}_delta_{size}_{time.strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    config = {
        "trial": trial,
        "model": args.model,
        "mode": "delta_residual",
        "size": size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_climate": n_clim,
        "climate_cols": climate_cols,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "device": str(device),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    best_score = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            x = batch["input"].to(device)
            st = batch["soft_t"].to(device)
            sf = batch["soft_f"].to(device)
            res = batch["residual"].to(device)
            c = batch["climate"].to(device)
            delta, soft_hat = model(x, st, c)
            loss = (
                F.l1_loss(delta, res)
                + F.l1_loss(soft_hat, sf)
                + 0.5 * (1.0 - dice_from_soft(soft_hat, sf)[0].mean())
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = loss_sum / max(n, 1)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        print(
            f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  "
            f"val_dice={val['dice']:.3f} (persist {val['dice_persist']:.3f})  "
            f"val_vegMAE={val['veg_mae']:.4f} (persist {val['veg_mae_persist']:.4f})  "
            f"val_softL1={val['soft_l1']:.4f} (persist {val['soft_l1_persist']:.4f})",
            flush=True,
        )
        # prefer higher dice & lower veg mae vs persist
        score = val["veg_mae"] + val["soft_l1"] - 0.5 * val["dice"]
        ckpt = {"epoch": epoch, "model": model.state_dict(), "config": config, "val": val}
        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
        if score < best_score:
            best_score = score
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            print(f"  saved best.pt  dice={val['dice']:.3f} vegMAE={val['veg_mae']:.4f}", flush=True)

        if epoch % 10 == 0 or epoch == 1:
            from torchvision.utils import save_image

            model.eval()
            batch = next(iter(val_loader))
            with torch.no_grad():
                _, soft_hat = model(
                    batch["input"].to(device),
                    batch["soft_t"].to(device),
                    batch["climate"].to(device),
                )
                soft_hat = soft_hat.cpu()
            rgb = (batch["input"][:4] + 1) * 0.5
            persist = batch["soft_t"][:4].repeat(1, 3, 1, 1)
            pr = soft_hat[:4].repeat(1, 3, 1, 1)
            gt = batch["soft_f"][:4].repeat(1, 3, 1, 1)
            save_image(torch.cat([rgb, persist, pr, gt], 0), run_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=4)

    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device)
    summary = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val": ckpt["val"],
        "test": test,
        "beats_persist_dice": test["dice"] > test["dice_persist"],
        "beats_persist_veg_mae": test["veg_mae"] < test["veg_mae_persist"],
        "beats_persist_soft_l1": test["soft_l1"] < test["soft_l1_persist"],
        "config": config,
        "run_dir": str(run_dir),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    with (run_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Done → {run_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", choices=["trial1", "trial2"], required=True)
    p.add_argument("--model", choices=["unet", "tinyt"], default="unet")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
