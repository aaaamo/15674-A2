import csv
import pathlib
import time
import numpy as np
import torch
from PIL import Image

from model import Camera, Optimizer2D, Optimizer3D

IMAGES = ["coffee", "astronaut", "cat"]
COUNTS = [256, 1024, 4096]
DENSIFY = [False, True]

N_3D = 4096
DENSIFY_3D = [True, False]

N_ORBIT = 16
ORBIT_ELEV = 25.0
ORBIT_TILT = 20.0

device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path("out")
RESULTS = OUT / "results.csv"


def load(name):
    img = Image.open(f"data/{name}.png").convert("RGB")
    return (torch.from_numpy(np.array(img)).float() / 255).to(device)


def psnr(img, target):
    return (-10 * torch.log10(((img - target) ** 2).mean())).item()


def save_result(name, tag, N, final_N, value, seconds):
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["image", "mode", "N", "final_N", "psnr", "time_s"])
        writer.writerow([name, tag, N, final_N, f"{value:.4f}", f"{seconds:.1f}"])


def test_2d():
    results = {}
    times = {}
    for N in COUNTS:
        for name in IMAGES:
            for D in DENSIFY:
                tag = "densify" if D else "fixed"
                print(f"== {name}  N={N}  ({tag})")
                target = load(name)
                torch.manual_seed(0)
                opt = Optimizer2D(target, budget=N)
                start = time.perf_counter()
                img = opt.optimize() if D else opt.optimize(N, densify=False)
                img = img.detach().clamp(0, 1)
                value = psnr(img, target)
                seconds = time.perf_counter() - start
                results[tag, name, N] = value
                times[tag, name, N] = seconds
                print(f"PSNR {value:.2f} dB  time {seconds:.1f} s")

                out = (torch.cat([target, img], dim=1) * 255).byte().cpu().numpy()
                Image.fromarray(out).save(OUT / f"{name}_{tag}_{N}.png")
                save_result(name, tag, N, opt.final_N, value, seconds)

                del opt, img
                torch.cuda.empty_cache()

    for title, table, fmt in [
        ("PSNR (dB)", results, "10.2f"),
        ("time (s)", times, "10.1f"),
    ]:
        for D in DENSIFY:
            tag = "densify" if D else "fixed"
            print(f"\n{title}, {tag}")
            print(f"{'image':10s}" + "".join(f"{f'N={N}':>10s}" for N in COUNTS))
            for name in IMAGES:
                row = "".join(f"{table[tag, name, N]:{fmt}}" for N in COUNTS)
                print(f"{name:10s}" + row)


def test_3d():
    for D in DENSIFY_3D:
        tag = "densify" if D else "fixed"
        print(f"== spheres  N={N_3D}  ({tag})")
        camera = Camera(pathlib.Path("data/spheres/cameras.json"))
        torch.manual_seed(0)
        opt = Optimizer3D(camera, budget=N_3D)
        start = time.perf_counter()
        if D:
            opt.optimize()
        else:
            opt.optimize(N_3D, densify=False)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        print(f"time {seconds:.1f} s")
        run_dir = OUT / f"spheres_{tag}_{N_3D}"
        run_dir.mkdir(exist_ok=True)
        for split, frames in [("train", camera.frames), ("val", camera.val_frames)]:
            with torch.no_grad():
                renders = [opt.gaussians(camera, fr).clamp(0, 1) for fr in frames]
            scores = [psnr(r, fr["img"]) for r, fr in zip(renders, frames)]
            value = sum(scores) / len(scores)
            print(f"{split} PSNR {value:.2f} dB over {len(scores)} views")
            save_result(f"spheres_{split}", tag, N_3D, opt.final_N, value, seconds)

            for i, (fr, img) in enumerate(zip(frames, renders)):
                out = (torch.cat([fr["img"], img], dim=1) * 255).byte()
                Image.fromarray(out.cpu().numpy()).save(
                    run_dir / f"{split}_{i:03d}.png"
                )

        poses = camera.orbit(N_ORBIT, ORBIT_ELEV, ORBIT_TILT)
        with torch.no_grad():
            renders = [opt.gaussians(camera, fr).clamp(0, 1) for fr in poses]
        orbit = [Image.fromarray((r * 255).byte().cpu().numpy()) for r in renders]
        for i, img in enumerate(orbit):
            img.save(run_dir / f"orbit_{i:03d}.png")
        orbit[0].save(
            run_dir / "orbit.gif",
            save_all=True,
            append_images=orbit[1:],
            duration=150,
            loop=0,
        )

        del opt, renders
        torch.cuda.empty_cache()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    test_2d()
    test_3d()
