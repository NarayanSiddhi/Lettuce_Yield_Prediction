#!/usr/bin/env python3
"""
Aim 2 Phase B (v2) — future canopy phenotype forecast with CNN / tiny Transformer.

Target is NOT RGB photos. We predict the t+4 ExGR vegetation mask from cup RGB@t
(+ climate), then score phenotype (Dice, IoU, veg-fraction MAE) vs persist.

Take: FGTD-style phenotype target; SegFormer-like patch attention (tiny).
Skip: Pix2Pix/ControlNet RGB retrain; full Agricrafter.
Improve: beat persist on mask metrics with U-Net and TinyTransformer.

Usage:
  python3 ml/forecast_images/train_aim2_phase_b_mask.py --trial trial2 --model unet --epochs 80
  python3 ml/forecast_images/train_aim2_phase_b_mask.py --trial trial2 --model tinyt --epochs 80
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
OUT_ROOT = IMG_DIR / "aim2_phase_b_mask"


def rgb_to_exgr_mask(img_chw01: np.ndarray, thr_pct: float = 60.0) -> np.ndarray:
    """CHW float [0,1] → binary mask HW float {0,1}."""
    r, g, b = img_chw01[0], img_chw01[1], img_chw01[2]
    exgr = 2.0 * g - r - b
    thr = float(np.percentile(exgr, thr_pct))
    return (exgr > max(thr, 0.02)).astype(np.float32)


class MaskPairDataset(Dataset):
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
        return arr.transpose(2, 0, 1)  # CHW [0,1]

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
        x_mask = rgb_to_exgr_mask(x01)
        y_mask = rgb_to_exgr_mask(y01)
        x = torch.from_numpy(x01 * 2.0 - 1.0)  # [-1,1] RGB
        return {
            "input": x,
            "input_mask": torch.from_numpy(x_mask).unsqueeze(0),
            "target_mask": torch.from_numpy(y_mask).unsqueeze(0),
            "climate": self._climate(row),
            "stem": Path(row["input_png"]).stem,
        }


# -------------------- models --------------------


def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetFutureMask(nn.Module):
    """Compact U-Net: RGB(+mask)+climate → future mask logits."""

    def __init__(self, n_climate: int = 12, base: int = 32):
        super().__init__()
        self.n_climate = n_climate
        in_ch = 3 + 1 + n_climate  # RGB + current mask + climate maps
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
        self.head = nn.Conv2d(b, 1, 1)

    def _fuse(self, x, mask, climate):
        b, _, h, w = x.shape
        c = climate.view(b, -1, 1, 1).expand(-1, -1, h, w)
        return torch.cat([x, mask, c], dim=1)

    def forward(self, x, mask, climate):
        z = self._fuse(x, mask, climate)
        e1 = self.enc1(z)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bot(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


class TinyTransformerFutureMask(nn.Module):
    """
    Tiny patch-Transformer encoder + CNN decoder (SegFormer-inspired, much smaller).
    Patches of RGB+mask; climate as extra tokens / FiLM on decoder.
    """

    def __init__(self, n_climate: int = 12, img_size: int = 128, patch: int = 8, dim: int = 128, depth: int = 4, heads: int = 4):
        super().__init__()
        self.n_climate = n_climate
        self.patch = patch
        self.dim = dim
        in_ch = 4  # RGB + current mask
        self.n_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.clim_fc = nn.Linear(n_climate, dim)
        # decode: reshape tokens → feature map → upsample
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
        )
        # patch=8 → side=16; ×2×2×2 = 128

    def forward(self, x, mask, climate):
        z = torch.cat([x, mask], dim=1)
        tok = self.proj(z)  # B,dim,S,S
        b, d, s, _ = tok.shape
        tok = tok.flatten(2).transpose(1, 2)  # B,N,D
        tok = tok + self.pos
        # add climate as bias to all tokens
        clim = self.clim_fc(climate).unsqueeze(1)
        tok = tok + clim
        tok = self.encoder(tok)
        feat = tok.transpose(1, 2).reshape(b, d, s, s)
        return self.dec(feat)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


def batch_metrics(logits, target, persist_mask):
    """Return dict of means over batch."""
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    tgt = target
    # Dice / IoU
    inter = (pred * tgt).sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + 1e-6)
    union = pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    veg_t = tgt.mean(dim=(1, 2, 3))
    veg_p = pred.mean(dim=(1, 2, 3))
    veg_persist = persist_mask.mean(dim=(1, 2, 3))
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "veg_mae": float((veg_p - veg_t).abs().mean().item()),
        "veg_mae_persist": float((veg_persist - veg_t).abs().mean().item()),
        "dice_persist": float(
            (
                (2 * (persist_mask * tgt).sum(dim=(1, 2, 3)) + 1e-6)
                / (persist_mask.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + 1e-6)
            )
            .mean()
            .item()
        ),
    }


def build_model(name: str, n_climate: int, size: int) -> nn.Module:
    if name == "unet":
        return UNetFutureMask(n_climate=n_climate)
    if name == "tinyt":
        return TinyTransformerFutureMask(n_climate=n_climate, img_size=size)
    raise SystemExit(f"Unknown model {name}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    agg = {k: 0.0 for k in ("dice", "iou", "veg_mae", "veg_mae_persist", "dice_persist", "loss")}
    n = 0
    for batch in loader:
        x = batch["input"].to(device)
        m = batch["input_mask"].to(device)
        y = batch["target_mask"].to(device)
        c = batch["climate"].to(device)
        logits = model(x, m, c)
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
        met = batch_metrics(logits, y, m)
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
    ds_root = IMG_DIR / "dataset" / f"{trial}_{size}"
    if not ds_root.exists():
        # fallback to 128 dataset
        ds_root = IMG_DIR / "dataset" / f"{trial}_128"
        size = 128
    blob = json.loads((ds_root / "climate_norm.json").read_text())
    climate_norm = blob["climate"]
    manifest = pd.read_csv(ds_root / "manifest.csv")
    climate_cols = sorted([c for c in climate_norm.keys() if c in manifest.columns])
    n_clim = len(climate_cols)

    def split_df(name):
        return manifest[manifest["split"] == name].reset_index(drop=True)

    train_ds = MaskPairDataset(split_df("train"), climate_cols, climate_norm, size)
    val_ds = MaskPairDataset(split_df("val"), climate_cols, climate_norm, size)
    test_ds = MaskPairDataset(split_df("test"), climate_cols, climate_norm, size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model, n_clim, size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_dir = OUT_ROOT / f"{trial}_{args.model}_{size}_{time.strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    config = {
        "trial": trial,
        "model": args.model,
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

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            x = batch["input"].to(device)
            m = batch["input_mask"].to(device)
            y = batch["target_mask"].to(device)
            c = batch["climate"].to(device)
            logits = model(x, m, c)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
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
            f"val_vegMAE={val['veg_mae']:.4f} (persist {val['veg_mae_persist']:.4f})",
            flush=True,
        )
        # early selection: minimize val veg MAE, tie-break by dice
        score = val["veg_mae"] - 0.01 * val["dice"]
        ckpt = {"epoch": epoch, "model": model.state_dict(), "config": config, "val": val}
        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
        if score < best_val:
            best_val = score
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            print(f"  saved best.pt  vegMAE={val['veg_mae']:.4f} dice={val['dice']:.3f}", flush=True)

        if epoch % 10 == 0 or epoch == 1:
            # save a few sample overlays
            model.eval()
            batch = next(iter(val_loader))
            with torch.no_grad():
                logits = model(batch["input"].to(device), batch["input_mask"].to(device), batch["climate"].to(device))
                pred = (torch.sigmoid(logits) > 0.5).float().cpu()
            # grid: input RGB | persist mask | pred | target
            from torchvision.utils import save_image

            rgb = (batch["input"][:4] + 1) * 0.5
            persist = batch["input_mask"][:4].repeat(1, 3, 1, 1)
            pr = pred[:4].repeat(1, 3, 1, 1)
            gt = batch["target_mask"][:4].repeat(1, 3, 1, 1)
            grid = torch.cat([rgb, persist, pr, gt], dim=0)
            save_image(grid, run_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=4)

    # Test on best
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device)
    summary = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val": ckpt["val"],
        "test": test,
        "beats_persist_dice": test["dice"] > test["dice_persist"],
        "beats_persist_veg_mae": test["veg_mae"] < test["veg_mae_persist"],
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
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", choices=["trial1", "trial2"], required=True)
    p.add_argument("--model", choices=["unet", "tinyt"], default="unet")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
