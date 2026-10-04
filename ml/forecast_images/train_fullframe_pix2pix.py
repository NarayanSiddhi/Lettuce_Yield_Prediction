#!/usr/bin/env python3
"""
Aim 2 — full-frame Pix2Pix on RAW grow-tent images (not segmented crops).

Input:  full-frame@t (+ sensors/actuators)
Target: full-frame@{t+horizon}

Pairs from build_fullframe_pairs.py (dense same-hour matching).

Usage:
  python3 ml/forecast_images/train_fullframe_pix2pix.py --trial trial1 --horizon 5 --size 256 --epochs 100
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[2]
IMG = Path(__file__).resolve().parent
OUT_ROOT = IMG / "runs_fullframe"


def conv_down(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm, padding_mode="reflect")]
    if norm:
        layers += [nn.BatchNorm2d(out_ch)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


def conv_up(in_ch, out_ch, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class GeneratorUNet(nn.Module):
    """8-level U-Net for 256×256 (also works for 128 if we resize inputs to 256)."""

    def __init__(self, n_climate: int = 0):
        super().__init__()
        c0 = 3 + n_climate
        self.d1 = conv_down(c0, 64, norm=False)
        self.d2 = conv_down(64, 128)
        self.d3 = conv_down(128, 256)
        self.d4 = conv_down(256, 512)
        self.d5 = conv_down(512, 512)
        self.d6 = conv_down(512, 512)
        self.d7 = conv_down(512, 512)
        self.d8 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=True, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        self.u1 = conv_up(512, 512, dropout=True)
        self.u2 = conv_up(1024, 512, dropout=True)
        self.u3 = conv_up(1024, 512, dropout=True)
        self.u4 = conv_up(1024, 512)
        self.u5 = conv_up(1024, 256)
        self.u6 = conv_up(512, 128)
        self.u7 = conv_up(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x, climate):
        b, _, h, w = x.shape
        if climate is not None and climate.numel():
            c = climate.view(b, -1, 1, 1).expand(-1, -1, h, w)
            x_in = torch.cat([x, c], 1)
        else:
            x_in = x
        d1 = self.d1(x_in)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))


class PatchDiscriminator(nn.Module):
    def __init__(self, n_climate: int = 0):
        super().__init__()
        c0 = 6 + n_climate  # x + y + climate
        self.net = nn.Sequential(
            conv_down(c0, 64, norm=False),
            conv_down(64, 128),
            conv_down(128, 256),
            nn.Conv2d(256, 512, 4, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect"),
        )
        self.n_climate = n_climate

    def forward(self, x, y, climate):
        b, _, h, w = x.shape
        if climate is not None and climate.numel():
            c = climate.view(b, -1, 1, 1).expand(-1, -1, h, w)
            z = torch.cat([x, y, c], 1)
        else:
            z = torch.cat([x, y], 1)
        return self.net(z)


class FullframeDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, feat_cols: list[str], norm: dict, size: int):
        self.df = manifest.reset_index(drop=True)
        self.feat_cols = feat_cols
        self.norm = norm
        self.size = size

    def __len__(self):
        return len(self.df)

    def _load(self, rel: str) -> torch.Tensor:
        img = Image.open(PROJECT / rel).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _climate(self, row) -> torch.Tensor:
        vals = []
        for c in self.feat_cols:
            v = float(pd.to_numeric(row.get(c, np.nan), errors="coerce"))
            if not np.isfinite(v):
                v = self.norm[c]["mean"]
            m, s = self.norm[c]["mean"], self.norm[c]["std"] or 1.0
            vals.append((v - m) / s)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "input": self._load(row["input_png"]),
            "target": self._load(row["target_png"]),
            "climate": self._climate(row),
            "stem": Path(row["input_png"]).stem,
        }


def ssim_simple(a, b):
    a = (a + 1) * 0.5
    b = (b + 1) * 0.5
    c1, c2 = 0.01**2, 0.03**2
    mu_a = a.mean(dim=(2, 3), keepdim=True)
    mu_b = b.mean(dim=(2, 3), keepdim=True)
    sig_a = ((a - mu_a) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_b = ((b - mu_b) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_ab = ((a - mu_a) * (b - mu_b)).mean(dim=(2, 3), keepdim=True)
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    )
    return float(ssim.mean().item())


def exgr_veg_frac(t: torch.Tensor) -> float:
    x = ((t.detach().cpu().numpy() + 1) * 0.5).clip(0, 1)
    r, g, b = x[0], x[1], x[2]
    exgr = 2 * g - r - b
    thr = float(np.percentile(exgr, 60))
    return float((exgr > max(thr, 0.02)).mean())


@torch.no_grad()
def eval_loader(G, loader, device, l1):
    G.eval()
    sum_l1 = sum_ssim = 0.0
    veg_p = veg_f = []
    n = 0
    persist_err = fake_err = 0.0
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        c = batch["climate"].to(device)
        fake = G(x, c)
        bs = x.size(0)
        sum_l1 += float(l1(fake, y).item()) * bs
        sum_ssim += ssim_simple(fake, y) * bs
        for i in range(bs):
            vi = exgr_veg_frac(x[i])
            vr = exgr_veg_frac(y[i])
            vf = exgr_veg_frac(fake[i])
            persist_err += abs(vi - vr)
            fake_err += abs(vf - vr)
        n += bs
    return {
        "l1": sum_l1 / max(n, 1),
        "ssim": sum_ssim / max(n, 1),
        "veg_mae_persist": persist_err / max(n, 1),
        "veg_mae_fake": fake_err / max(n, 1),
        "n": n,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds_root = IMG / "dataset" / f"fullframe_{args.trial}_h{args.horizon}_{args.size}"
    if not ds_root.exists():
        raise SystemExit(f"Missing dataset {ds_root}. Run build_fullframe_pairs.py first.")
    meta = json.loads((ds_root / "climate_norm.json").read_text())
    feat_cols = meta["feature_cols"]
    norm = meta["climate"]
    man = pd.read_csv(ds_root / "manifest.csv")
    # drop rows without exported images
    man = man[man["input_png"].notna() & man["target_png"].notna()].reset_index(drop=True)

    train_ds = FullframeDataset(man[man.split == "train"], feat_cols, norm, args.size)
    val_ds = FullframeDataset(man[man.split == "val"], feat_cols, norm, args.size)
    test_ds = FullframeDataset(man[man.split == "test"], feat_cols, norm, args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    n_clim = len(feat_cols)
    G = GeneratorUNet(n_climate=n_clim).to(device)
    D = PatchDiscriminator(n_climate=n_clim).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    run_dir = OUT_ROOT / f"{args.trial}_h{args.horizon}_{args.size}_{time.strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    config = {
        "trial": args.trial,
        "horizon": args.horizon,
        "size": args.size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_l1": args.lambda_l1,
        "n_climate": n_clim,
        "feature_cols": feat_cols,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "device": str(device),
        "source": "fullframe_raw_not_segmented",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        g_sum = d_sum = 0.0
        n = 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            c = batch["climate"].to(device)
            bs = x.size(0)
            # --- D ---
            with torch.no_grad():
                fake = G(x, c)
            real_logits = D(x, y, c)
            fake_logits = D(x, fake.detach(), c)
            loss_D = 0.5 * (
                bce(real_logits, torch.ones_like(real_logits))
                + bce(fake_logits, torch.zeros_like(fake_logits))
            )
            optD.zero_grad(set_to_none=True)
            loss_D.backward()
            optD.step()
            # --- G ---
            fake = G(x, c)
            fake_logits = D(x, fake, c)
            loss_G = bce(fake_logits, torch.ones_like(fake_logits)) + args.lambda_l1 * l1(fake, y)
            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()
            g_sum += float(loss_G.item()) * bs
            d_sum += float(loss_D.item()) * bs
            n += bs

        val = eval_loader(G, val_loader, device, l1)
        history.append(
            {
                "epoch": epoch,
                "loss_G": g_sum / max(n, 1),
                "loss_D": d_sum / max(n, 1),
                "val_l1": val["l1"],
                "val_ssim": val["ssim"],
                "val_veg_mae_fake": val["veg_mae_fake"],
                "val_veg_mae_persist": val["veg_mae_persist"],
            }
        )
        print(
            f"epoch {epoch:03d}/{args.epochs}  G={g_sum/max(n,1):.3f} D={d_sum/max(n,1):.3f}  "
            f"val_L1={val['l1']:.4f} SSIM={val['ssim']:.3f}  "
            f"vegMAE={val['veg_mae_fake']:.4f} (persist {val['veg_mae_persist']:.4f})",
            flush=True,
        )
        ckpt = {"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(), "config": config, "val": val}
        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
        if val["l1"] < best_val:
            best_val = val["l1"]
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            print(f"  saved best.pt val_L1={val['l1']:.4f}", flush=True)

        if epoch % 10 == 0 or epoch == 1:
            from torchvision.utils import save_image

            G.eval()
            batch = next(iter(val_loader))
            with torch.no_grad():
                fake = G(batch["input"].to(device), batch["climate"].to(device)).cpu()
            grid = torch.cat([batch["input"][:4], fake[:4], batch["target"][:4]], 0)
            save_image(grid, run_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=4, normalize=True, value_range=(-1, 1))

    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G"])
    test = eval_loader(G, test_loader, device, l1)
    summary = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val": ckpt["val"],
        "test": test,
        "beats_persist_veg": test["veg_mae_fake"] < test["veg_mae_persist"],
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
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
