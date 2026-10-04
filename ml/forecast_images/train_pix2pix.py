"""
Phase A: Pix2Pix future-plant generator (Trial 1 or Trial 2 cups).

Input:  cup crop at DAT t  + climate vector
Target: cup crop at DAT t+4

Usage:
  python3 ml/forecast_images/build_pix2pix_pairs.py --trial trial2 --size 128 --export-images
  python3 ml/forecast_images/train_pix2pix.py --trial trial2 --epochs 300 --batch-size 8 --size 128
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


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
    """
    7-level U-Net for 128x128 (and 256x256):
      128→64→32→16→8→4→2→1 then up symmetrically.
    Climate is tiled as extra input channels.
    """

    def __init__(self, n_climate: int = 0):
        super().__init__()
        self.n_climate = n_climate
        c0 = 3 + n_climate
        self.d1 = conv_down(c0, 64, norm=False)   # /2
        self.d2 = conv_down(64, 128)              # /4
        self.d3 = conv_down(128, 256)             # /8
        self.d4 = conv_down(256, 512)             # /16
        self.d5 = conv_down(512, 512)             # /32
        self.d6 = conv_down(512, 512)             # /64
        self.d7 = conv_down(512, 512)             # /128
        self.u1 = conv_up(512, 512, dropout=True)
        self.u2 = conv_up(1024, 512, dropout=True)
        self.u3 = conv_up(1024, 512, dropout=True)
        self.u4 = conv_up(1024, 512)
        self.u5 = conv_up(1024, 256)
        self.u6 = conv_up(512, 128)
        self.u7 = conv_up(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def _tile_climate(self, x, climate):
        if self.n_climate == 0:
            return x
        b, _, h, w = x.shape
        clim = climate.view(b, self.n_climate, 1, 1).expand(b, self.n_climate, h, w)
        return torch.cat([x, clim], dim=1)

    def forward(self, x, climate: torch.Tensor | None = None):
        x = self._tile_climate(x, climate)
        d1 = self.d1(x)
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
        u7 = self.u7(torch.cat([u6, d1], 1))
        return self.final(torch.cat([u7, x[:, :3]], 1))  # skip with RGB only; wrong ch if climate

    # Fix final skip: should skip with d1-compatible path. Use d1 skip only via u7 already.
    # Override forward properly without broken final skip.


class GeneratorUNetFixed(nn.Module):
    def __init__(self, n_climate: int = 0):
        super().__init__()
        self.n_climate = n_climate
        c0 = 3 + n_climate
        self.d1 = conv_down(c0, 64, norm=False)  # 64 @ H/2
        self.d2 = conv_down(64, 128)             # 128 @ H/4
        self.d3 = conv_down(128, 256)            # 256 @ H/8
        self.d4 = conv_down(256, 512)            # 512 @ H/16
        self.d5 = conv_down(512, 512)            # 512 @ H/32
        self.d6 = conv_down(512, 512)            # 512 @ H/64
        self.d7 = nn.Sequential(                 # 512 @ H/128
            nn.Conv2d(512, 512, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        self.u1 = conv_up(512, 512, dropout=True)     # -> cat d6 = 1024
        self.u2 = conv_up(1024, 512, dropout=True)    # -> cat d5 = 1024
        self.u3 = conv_up(1024, 512, dropout=True)    # -> cat d4 = 1024
        self.u4 = conv_up(1024, 256)                  # -> cat d3 = 512
        self.u5 = conv_up(512, 128)                   # -> cat d2 = 256
        self.u6 = conv_up(256, 64)                    # -> cat d1 = 128
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x, climate: torch.Tensor | None = None):
        if self.n_climate > 0:
            assert climate is not None
            b, _, h, w = x.shape
            clim = climate.view(b, self.n_climate, 1, 1).expand(b, self.n_climate, h, w)
            x_in = torch.cat([x, clim], dim=1)
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


class Discriminator(nn.Module):
    def __init__(self, n_climate: int = 0):
        super().__init__()
        self.n_climate = n_climate
        c_in = 6 + n_climate
        self.model = nn.Sequential(
            conv_down(c_in, 64, norm=False),
            conv_down(64, 128),
            conv_down(128, 256),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x, y, climate: torch.Tensor | None = None):
        xy = torch.cat([x, y], 1)
        if self.n_climate > 0:
            assert climate is not None
            b, _, h, w = xy.shape
            clim = climate.view(b, self.n_climate, 1, 1).expand(b, self.n_climate, h, w)
            xy = torch.cat([xy, clim], 1)
        return self.model(xy)


class PairDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, climate_cols: list[str], norm: dict, size: int):
        self.df = manifest.reset_index(drop=True)
        self.climate_cols = climate_cols
        self.norm = norm
        self.size = size

    def __len__(self):
        return len(self.df)

    def _load(self, rel: str) -> torch.Tensor:
        img = Image.open(PROJECT / rel).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

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
        return {
            "input": self._load(row["input_png"]),
            "target": self._load(row["target_png"]),
            "climate": self._climate(row),
            "stem": Path(row["input_png"]).stem,
        }


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a + 1) * 0.5
    b = (b + 1) * 0.5
    c1, c2 = 0.01**2, 0.03**2
    mu_a = a.mean(dim=(2, 3), keepdim=True)
    mu_b = b.mean(dim=(2, 3), keepdim=True)
    sig_a = ((a - mu_a) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_b = ((b - mu_b) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_ab = ((a - mu_a) * (b - mu_b)).mean(dim=(2, 3), keepdim=True)
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2))
    return float(ssim.mean().item())


def save_grid(path: Path, tensors: list[torch.Tensor], nrow: int = 3) -> None:
    try:
        from torchvision.utils import make_grid, save_image
    except ImportError:
        # fallback: save first triple only via PIL
        path.parent.mkdir(parents=True, exist_ok=True)
        imgs = []
        for t in tensors[:9]:
            arr = ((t.clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).cpu().numpy()
            imgs.append(Image.fromarray(arr))
        w, h = imgs[0].size
        ncols = 3
        nrows = (len(imgs) + ncols - 1) // ncols
        canvas = Image.new("RGB", (ncols * w, nrows * h))
        for i, im in enumerate(imgs):
            canvas.paste(im, ((i % ncols) * w, (i // ncols) * h))
        canvas.save(path)
        return
    grid = make_grid(tensors, nrow=nrow, normalize=True, value_range=(-1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(path))


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    trial = args.trial
    trial_tag = "t1" if trial == "trial1" else "t2"
    ds_root = OUT_DIR / "dataset" / f"{trial}_{args.size}"
    manifest_path = ds_root / "manifest.csv"
    norm_path = ds_root / "climate_norm.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}. Run:\n"
            f"  python3 ml/forecast_images/build_pix2pix_pairs.py --trial {trial} "
            f"--size {args.size} --export-images"
        )

    if args.size not in (128, 256):
        raise SystemExit("--size must be 128 or 256 for this U-Net")

    manifest = pd.read_csv(manifest_path)
    norm_blob = json.loads(norm_path.read_text())
    climate_cols = sorted(norm_blob["climate"].keys())
    n_clim = len(climate_cols)

    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]
    if len(test_df) == 0:
        raise SystemExit(
            "No test split in manifest. Rebuild with:\n"
            f"  python3 ml/forecast_images/build_pix2pix_pairs.py --trial {trial} "
            f"--size {args.size} --export-images"
        )
    if len(train_df) < args.batch_size:
        args.batch_size = max(1, len(train_df))

    train_ds = PairDataset(train_df, climate_cols, norm_blob["climate"], args.size)
    val_ds = PairDataset(val_df, climate_cols, norm_blob["climate"], args.size)
    test_ds = PairDataset(test_df, climate_cols, norm_blob["climate"], args.size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=len(train_ds) >= args.batch_size
    )
    val_loader = DataLoader(val_ds, batch_size=min(4, max(1, len(val_ds))), shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=min(4, max(1, len(test_ds))), shuffle=False, num_workers=1)

    G = GeneratorUNetFixed(n_climate=n_clim).to(device)
    D = Discriminator(n_climate=n_clim).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    run_dir = OUT_DIR / "runs" / f"pix2pix_{trial_tag}_{args.size}_{time.strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["epoch", "loss_G", "loss_D", "val_l1", "val_ssim"]).writeheader()

    config = {
        "trial": trial,
        "size": args.size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_l1": args.lambda_l1,
        "n_climate": n_clim,
        "climate_cols": climate_cols,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "train_cups": norm_blob.get("train_cups"),
        "val_cups": norm_blob.get("val_cups"),
        "test_cups": norm_blob.get("test_cups"),
        "device": str(device),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    # shape smoke check
    with torch.no_grad():
        xb = torch.zeros(1, 3, args.size, args.size, device=device)
        cb = torch.zeros(1, n_clim, device=device)
        yb = G(xb, cb)
        assert yb.shape == xb.shape, yb.shape

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        loss_G_epoch = 0.0
        loss_D_epoch = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            c = batch["climate"].to(device)

            with torch.no_grad():
                fake = G(x, c)
            pred_real = D(x, y, c)
            pred_fake = D(x, fake.detach(), c)
            loss_D = 0.5 * (
                bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake))
            )
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            fake = G(x, c)
            pred_fake = D(x, fake, c)
            loss_G = bce(pred_fake, torch.ones_like(pred_fake)) + args.lambda_l1 * l1(fake, y)
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            loss_G_epoch += float(loss_G.item())
            loss_D_epoch += float(loss_D.item())
            n_batches += 1

        G.eval()
        val_l1 = 0.0
        val_ssim = 0.0
        n_val = 0
        sample_tensors: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                c = batch["climate"].to(device)
                fake = G(x, c)
                val_l1 += float(l1(fake, y).item()) * x.size(0)
                val_ssim += ssim_simple(fake, y) * x.size(0)
                n_val += x.size(0)
                for i in range(x.size(0)):
                    if len(sample_tensors) >= 9:
                        break
                    sample_tensors.extend([x[i].cpu(), fake[i].cpu(), y[i].cpu()])
        val_l1 /= max(n_val, 1)
        val_ssim /= max(n_val, 1)
        loss_G_epoch /= max(n_batches, 1)
        loss_D_epoch /= max(n_batches, 1)

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["epoch", "loss_G", "loss_D", "val_l1", "val_ssim"]).writerow(
                {
                    "epoch": epoch,
                    "loss_G": f"{loss_G_epoch:.4f}",
                    "loss_D": f"{loss_D_epoch:.4f}",
                    "val_l1": f"{val_l1:.4f}",
                    "val_ssim": f"{val_ssim:.4f}",
                }
            )

        print(
            f"epoch {epoch:03d}/{args.epochs}  G={loss_G_epoch:.3f}  D={loss_D_epoch:.3f}  "
            f"val_L1={val_l1:.4f}  val_SSIM={val_ssim:.4f}",
            flush=True,
        )

        if epoch % args.sample_every == 0 or epoch == 1 or epoch == args.epochs:
            if sample_tensors:
                save_grid(run_dir / "samples" / f"epoch_{epoch:03d}.png", sample_tensors, nrow=3)

        ckpt = {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "config": config,
            "val_l1": val_l1,
            "val_ssim": val_ssim,
        }
        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
        if val_l1 < best_val:
            best_val = val_l1
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            print(f"  saved best.pt (val_L1={val_l1:.4f})", flush=True)

    # Final test evaluation on best checkpoint only (once)
    best_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G"])
    G.eval()
    test_l1 = 0.0
    test_ssim = 0.0
    n_test = 0
    test_tensors: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            c = batch["climate"].to(device)
            fake = G(x, c)
            test_l1 += float(l1(fake, y).item()) * x.size(0)
            test_ssim += ssim_simple(fake, y) * x.size(0)
            n_test += x.size(0)
            for i in range(x.size(0)):
                if len(test_tensors) >= 9:
                    break
                test_tensors.extend([x[i].cpu(), fake[i].cpu(), y[i].cpu()])
    test_l1 /= max(n_test, 1)
    test_ssim /= max(n_test, 1)
    test_summary = {
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_l1": float(ckpt.get("val_l1", best_val)),
        "best_val_ssim": float(ckpt.get("val_ssim", float("nan"))),
        "test_l1": test_l1,
        "test_ssim": test_ssim,
        "n_test": n_test,
        "test_cups": norm_blob.get("test_cups"),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(test_summary, indent=2))
    if test_tensors:
        save_grid(run_dir / "samples" / "test_grid.png", test_tensors, nrow=3)
    print(json.dumps(test_summary, indent=2), flush=True)
    print(f"Done. Outputs in {run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=["trial1", "trial2"], default="trial1")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
