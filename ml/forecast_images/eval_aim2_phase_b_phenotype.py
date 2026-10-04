#!/usr/bin/env python3
"""
Aim 2 Phase B — phenotype consistency on existing Pix2Pix checkpoints.

Loads legacy Generator matching saved best.pt (arch differs from current train_pix2pix.py).
Scores ExGR vegetation-fraction MAE (fake vs real) vs persist (input vs real).
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

IMG_DIR = Path(__file__).resolve().parent
OUT_DIR = IMG_DIR / "aim2_phase_b"

RUNS = {
    "trial1": IMG_DIR / "runs" / "pix2pix_t1_128_20260908_105032",
    "trial2": IMG_DIR / "runs" / "pix2pix_t2_128_20260908_130245",
}


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


class LegacyGeneratorUNet(nn.Module):
    """Matches pix2pix best.pt from Sep 2026 runs (d1–d7, u1–u6, final)."""

    def __init__(self, n_climate: int = 12):
        super().__init__()
        c0 = 3 + n_climate
        self.d1 = conv_down(c0, 64, norm=False)
        self.d2 = conv_down(64, 128)
        self.d3 = conv_down(128, 256)
        self.d4 = conv_down(256, 512)
        self.d5 = conv_down(512, 512)
        self.d6 = conv_down(512, 512)
        # bottleneck: no BatchNorm (matches ckpt d7.0.bias)
        self.d7 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=True, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        self.u1 = conv_up(512, 512, dropout=True)
        self.u2 = conv_up(1024, 512, dropout=True)
        self.u3 = conv_up(1024, 512, dropout=True)
        self.u4 = conv_up(1024, 256)
        self.u5 = conv_up(512, 128)
        self.u6 = conv_up(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x, climate):
        b, _, h, w = x.shape
        if climate is not None and climate.numel() > 0:
            c = climate.view(b, -1, 1, 1).expand(-1, -1, h, w)
            x_in = torch.cat([x, c], dim=1)
        else:
            x_in = x
        d1 = self.d1(x_in)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        u1 = self.u1(d7)
        u2 = self.u2(torch.cat([u1, d6], 1))
        u3 = self.u3(torch.cat([u2, d5], 1))
        u4 = self.u4(torch.cat([u3, d4], 1))
        u5 = self.u5(torch.cat([u4, d3], 1))
        u6 = self.u6(torch.cat([u5, d2], 1))
        return self.final(torch.cat([u6, d1], 1))


def exgr_veg_frac(img_chw: np.ndarray) -> float:
    x = img_chw.astype(np.float32)
    if x.min() < -0.01:
        x = (x + 1.0) * 0.5
    x = np.clip(x, 0, 1)
    r, g, b = x[0], x[1], x[2]
    exgr = 2.0 * g - r - b
    thr = float(np.percentile(exgr, 60))
    return float((exgr > max(thr, 0.02)).mean())


def eval_trial(trial: str, device: str) -> tuple[dict, list]:
    api = runpy.run_path(str(IMG_DIR / "train_pix2pix.py"))
    run_dir = RUNS[trial]
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    n_clim = int(ckpt["config"].get("n_climate", 12))
    G = LegacyGeneratorUNet(n_climate=n_clim).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    ds_root = IMG_DIR / "dataset" / f"{trial}_128"
    blob = json.loads((ds_root / "climate_norm.json").read_text())
    climate_norm = blob.get("climate", blob)
    manifest = pd.read_csv(ds_root / "manifest.csv")
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    climate_cols = list(ckpt["config"].get("climate_cols") or list(climate_norm.keys()))
    climate_cols = [c for c in climate_cols if c in climate_norm and c in test_rows.columns]
    if len(climate_cols) != n_clim:
        # fall back to sorted keys matching n_clim
        climate_cols = [c for c in sorted(climate_norm.keys()) if c in test_rows.columns][:n_clim]
    print(f"  climate_cols={len(climate_cols)} expected={n_clim}")
    test_ds = api["PairDataset"](test_rows, climate_cols, climate_norm, size=128)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    # sanity
    sample = test_ds[0]
    print(f"  sample input={tuple(sample['input'].shape)} climate={tuple(sample['climate'].shape)}")
    l1_fn = nn.L1Loss()

    rows = []
    sum_l1 = sum_ssim = 0.0
    persist_errs, fake_errs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            c = batch["climate"].to(device)
            fake = G(x, c)
            sum_l1 += float(l1_fn(fake, y).item())
            sum_ssim += float(api["ssim_simple"](fake, y))
            xin = x[0].cpu().numpy()
            yr = y[0].cpu().numpy()
            yf = fake[0].cpu().numpy()
            vi, vr, vf = exgr_veg_frac(xin), exgr_veg_frac(yr), exgr_veg_frac(yf)
            persist_errs.append(abs(vi - vr))
            fake_errs.append(abs(vf - vr))
            meta = test_rows.iloc[i].to_dict() if i < len(test_rows) else {}
            rows.append(
                {
                    "trial": trial,
                    "idx": i,
                    "cup_id": meta.get("cup_id", ""),
                    "day_t": meta.get("day_t", ""),
                    "day_future": meta.get("day_future", ""),
                    "l1": float(l1_fn(fake, y).item()),
                    "veg_frac_input": vi,
                    "veg_frac_real": vr,
                    "veg_frac_fake": vf,
                    "veg_abs_err_persist": abs(vi - vr),
                    "veg_abs_err_fake": abs(vf - vr),
                }
            )
    n = max(len(rows), 1)
    summary = {
        "trial": trial,
        "run": run_dir.name,
        "n_test": len(rows),
        "test_l1": sum_l1 / n,
        "test_ssim": sum_ssim / n,
        "veg_frac_mae_persist": float(np.mean(persist_errs)),
        "veg_frac_mae_fake": float(np.mean(fake_errs)),
        "veg_beats_persist": float(np.mean(persist_errs) - np.mean(fake_errs)),
    }
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", nargs="+", default=["trial1", "trial2"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Aim 2 Phase B  device={args.device}")
    summaries, all_rows = [], []
    for trial in args.trials:
        print(f"\n=== {trial} ===")
        s, rows = eval_trial(trial, args.device)
        summaries.append(s)
        all_rows.extend(rows)
        print(json.dumps(s, indent=2))
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "phase_b_per_image.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "phase_b_summary.csv", index=False)
    (OUT_DIR / "phase_b_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
