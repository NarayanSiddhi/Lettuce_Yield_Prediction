"""
Phase B: pretrained Stable Diffusion + ControlNet fine-tune (per trial).

Conditioning image = preprocessed cup at DAT t
Target image       = preprocessed cup at DAT t+4
Prompt             = lettuce + climate summary

Only ControlNet weights are trained; SD UNet/VAE/text encoder stay frozen.

Usage:
  python3 ml/forecast_images/preprocess_and_export.py --trial trial1 --size 256
  python3 ml/forecast_images/train_controlnet.py --trial trial1 --size 256 --epochs 300 --eval-every-epochs 10
  python3 ml/forecast_images/train_controlnet.py --trial trial2 --size 256 --epochs 300 --eval-every-epochs 10
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

DEFAULT_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class PairDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, size: int):
        self.df = manifest.reset_index(drop=True)
        self.size = size

    def __len__(self) -> int:
        return len(self.df)

    def _load_target(self, rel: str) -> torch.Tensor:
        img = Image.open(PROJECT / rel).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0  # [-1,1] for VAE
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_cond(self, rel: str) -> torch.Tensor:
        img = Image.open(PROJECT / rel).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1] ControlNet RGB control
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        r = self.df.iloc[idx]
        return {
            "conditioning": self._load_cond(r["conditioning_png"]),
            "target": self._load_target(r["target_png"]),
            "prompt": str(r["prompt"]),
        }


def ssim_simple(x: torch.Tensor, y: torch.Tensor) -> float:
    # x,y in [-1,1], NCHW
    x = (x + 1) * 0.5
    y = (y + 1) * 0.5
    c1, c2 = 0.01**2, 0.03**2
    mu_x = x.mean(dim=(1, 2, 3), keepdim=True)
    mu_y = y.mean(dim=(1, 2, 3), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(1, 2, 3), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim.mean().item())


@torch.no_grad()
def generate_batch(pipe, conditioning: torch.Tensor, prompts: list[str], steps: int, guidance: float):
    # conditioning in [0,1] NCHW → PIL
    imgs = []
    for i in range(conditioning.size(0)):
        t = conditioning[i].clamp(0, 1)
        arr = (t * 255.0).byte().cpu().permute(1, 2, 0).numpy()
        imgs.append(Image.fromarray(arr))
    out = pipe(
        prompt=list(prompts),
        image=imgs,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device=pipe.device).manual_seed(0),
    ).images
    tensors = []
    for im in out:
        arr = np.asarray(im.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def evaluate(pipe, loader, device, steps: int, guidance: float):
    l1s, ssims = [], []
    sample_tensors: list[torch.Tensor] = []
    for batch in loader:
        cond = batch["conditioning"].to(device)
        tgt = batch["target"]
        prompts = batch["prompt"]
        fake = generate_batch(pipe, cond, prompts, steps=steps, guidance=guidance).cpu()
        l1s.append(float(F.l1_loss(fake, tgt).item()))
        ssims.append(ssim_simple(fake, tgt))
        for i in range(cond.size(0)):
            if len(sample_tensors) >= 9:
                break
            cond_vis = cond[i].detach().cpu() * 2.0 - 1.0
            sample_tensors.extend([cond_vis, fake[i], tgt[i]])
    return float(np.mean(l1s)), float(np.mean(ssims)), sample_tensors


def train(args) -> None:
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        StableDiffusionControlNetPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    trial = args.trial
    trial_tag = "t1" if trial == "trial1" else "t2"
    ds_root = OUT_DIR / "dataset" / f"{trial}_controlnet_{args.size}"
    manifest_path = ds_root / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}. Run:\n"
            f"  python3 ml/forecast_images/preprocess_and_export.py --trial {trial} --size {args.size}"
        )

    manifest = pd.read_csv(manifest_path)
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]
    if len(train_df) == 0 or len(test_df) == 0:
        raise SystemExit("Need non-empty train and test splits.")

    train_ds = PairDataset(train_df, args.size)
    val_ds = PairDataset(val_df, args.size)
    test_ds = PairDataset(test_df, args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=min(2, max(1, len(val_ds))), shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=min(2, max(1, len(test_ds))), shuffle=False, num_workers=1)

    steps_per_epoch = max(1, len(train_loader))
    if args.epochs is not None:
        args.max_steps = int(args.epochs) * steps_per_epoch
        print(
            f"Using --epochs {args.epochs} → {args.max_steps} steps "
            f"({steps_per_epoch} steps/epoch, n_train={len(train_ds)}, batch={args.batch_size})",
            flush=True,
        )
    if args.eval_every_epochs is not None:
        args.eval_every = max(1, int(args.eval_every_epochs) * steps_per_epoch)
        print(f"Using --eval-every-epochs {args.eval_every_epochs} → eval every {args.eval_every} steps", flush=True)

    # fp32 is more reliable for ControlNet train+eval on A5000; use --fp16 to save VRAM.
    dtype = torch.float16 if (device.type == "cuda" and args.fp16) else torch.float32
    print(f"Loading SD + ControlNet from {args.pretrained} ({dtype})…", flush=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder").to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained, subfolder="vae").to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained, subfolder="unet").to(device, dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    # RGB image conditioning: init ControlNet from UNet
    controlnet = ControlNetModel.from_unet(unet)
    controlnet.to(device, dtype=dtype)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr, weight_decay=1e-2)

    run_dir = OUT_DIR / "runs" / f"controlnet_{trial_tag}_{args.size}_{time.strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["step", "loss", "val_l1", "val_ssim"]).writeheader()

    norm_blob = json.loads((ds_root / "climate_norm.json").read_text())
    config = {
        "trial": trial,
        "size": args.size,
        "pretrained": args.pretrained,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "train_cups": norm_blob.get("train_cups"),
        "val_cups": norm_blob.get("val_cups"),
        "test_cups": norm_blob.get("test_cups"),
        "device": str(device),
        "dtype": str(dtype),
        "preprocess": norm_blob.get("preprocess"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    # inference pipe (shares modules)
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    best_val = math.inf
    step = 0
    train_iter = iter(train_loader)
    loss_ema = None

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        pixel_values = batch["target"].to(device, dtype=dtype)
        conditioning = batch["conditioning"].to(device, dtype=dtype)
        prompts = batch["prompt"]

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            text_inputs = tokenizer(
                list(prompts),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]
            encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        down_samples, mid_sample = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=conditioning,
            return_dict=False,
        )
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=list(down_samples),
            mid_block_additional_residual=mid_sample,
        ).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(noise_scheduler.config.prediction_type)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        optimizer.step()

        step += 1
        loss_f = float(loss.item())
        loss_ema = loss_f if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_f

        do_eval = (step % args.eval_every == 0) or (step == args.max_steps) or (step == 1)
        if do_eval:
            controlnet.eval()
            val_l1, val_ssim, sample_tensors = evaluate(
                pipe, val_loader, device, steps=args.infer_steps, guidance=args.guidance
            )
            controlnet.train()
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=["step", "loss", "val_l1", "val_ssim"]).writerow(
                    {
                        "step": step,
                        "loss": f"{loss_ema:.4f}",
                        "val_l1": f"{val_l1:.4f}",
                        "val_ssim": f"{val_ssim:.4f}",
                    }
                )
            print(
                f"step {step:05d}/{args.max_steps}  loss={loss_ema:.4f}  "
                f"val_L1={val_l1:.4f}  val_SSIM={val_ssim:.4f}",
                flush=True,
            )
            if sample_tensors:
                grid = make_grid(torch.stack(sample_tensors), nrow=3, normalize=True, value_range=(-1, 1))
                save_image(grid, str(run_dir / "samples" / f"step_{step:05d}.png"))

            ckpt = {
                "step": step,
                "controlnet": controlnet.state_dict(),
                "config": config,
                "val_l1": val_l1,
                "val_ssim": val_ssim,
            }
            torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
            if val_l1 < best_val:
                best_val = val_l1
                torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
                controlnet.save_pretrained(run_dir / "checkpoints" / "best_controlnet")
                print(f"  saved best (val_L1={val_l1:.4f})", flush=True)
        elif step % 50 == 0:
            print(f"step {step:05d}/{args.max_steps}  loss={loss_ema:.4f}", flush=True)

    # Final test on best
    best_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    controlnet.load_state_dict(ckpt["controlnet"])
    controlnet.eval()
    test_l1, test_ssim, test_tensors = evaluate(
        pipe, test_loader, device, steps=args.infer_steps, guidance=args.guidance
    )
    summary = {
        "best_step": int(ckpt.get("step", -1)),
        "best_val_l1": float(ckpt.get("val_l1", best_val)),
        "best_val_ssim": float(ckpt.get("val_ssim", float("nan"))),
        "test_l1": test_l1,
        "test_ssim": test_ssim,
        "n_test": len(test_ds),
        "test_cups": norm_blob.get("test_cups"),
        "compare_note": "Compare against Pix2Pix runs with same cup split; images are preprocessed 256px here.",
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    if test_tensors:
        grid = make_grid(torch.stack(test_tensors), nrow=3, normalize=True, value_range=(-1, 1))
        save_image(grid, str(run_dir / "samples" / "test_grid.png"))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Done. Outputs in {run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=["trial1", "trial2"], required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pretrained", type=str, default=DEFAULT_SD)
    parser.add_argument("--epochs", type=int, default=None, help="If set, overrides --max-steps (fair vs Pix2Pix).")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=None,
        help="If set, overrides --eval-every (e.g. 10 ≈ Pix2Pix --sample-every 10).",
    )
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--fp16", action="store_true", help="Use float16 (faster; may hit dtype bugs).")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
