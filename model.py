from __future__ import annotations
import json
import math
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    # TODO: build R(theta) and S = diag(scale), return Sigma = R S S^T R^T  -> (N, 2, 2)
    S = torch.diag_embed(scale)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    R = torch.stack([cos, -sin, sin, cos], dim=-1).view(-1, 2, 2)
    Sigma = R @ S @ S.mT @ R.mT
    return Sigma


def gaussian_weight(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # TODO: w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
    # xymu = xy[:, None, :] - mu[None, :, :]  # (P, N, 2)
    # Sigma_inv = torch.linalg.inv(Sigma)
    # m = torch.einsum("pni,nij,pnj->pn", xymu, Sigma_inv, xymu)
    dx = xy[:, None, 0] - mu[None, :, 0]
    dy = xy[:, None, 1] - mu[None, :, 1]
    a, b, c = Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-8)
    m = ((c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det).clamp(min=0)
    return torch.exp(-0.5 * m)  # (P, N)


def pixel_grid(H, W):
    ys, xs = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)


def render(mu, Sigma, color, opacity, order, H, W, xy=None):
    # color: (N, 3),  opacity: (N,) in [0, 1],  order: indices sorted front -> back
    whole = xy is None
    if whole:
        xy = pixel_grid(H, W).to(mu.device)  # (H*W, 2)
    mu, Sigma, color, opacity = mu[order], Sigma[order], color[order], opacity[order]
    w = gaussian_weight(xy, mu, Sigma)  # (P, N)  from P1
    # alpha = opacity[None, :] * w  # (P, N)
    # TODO: C and T compositing here
    # C = torch.zeros(H * W, 3, device=mu.device)
    # T = torch.ones(H * W, device=mu.device)
    # for i in order:  # front to back
    #     a = alpha[:, i]  # (P,)
    #     C += (a * T)[:, None] * color[i][None, :]
    #     T = T * (1 - a)
    T = torch.cumprod(1 - opacity[None, :] * w, dim=1)
    dcolor = torch.cat([color[1:], torch.zeros_like(color[:1])]) - color
    C = color[0] + T @ dcolor
    return C.reshape(H, W, 3) if whole else C


class Gaussian2D(nn.Module):
    def __init__(self, mu, log_s, theta, color, op_raw):
        super().__init__()
        self.mu = nn.Parameter(mu.detach())
        self.log_s = nn.Parameter(log_s.detach())
        self.theta = nn.Parameter(theta.detach())
        self.color = nn.Parameter(color.detach())
        self.op_raw = nn.Parameter(op_raw.detach())

    @classmethod
    def random(cls, N, H, W) -> Gaussian2D:
        mu = torch.rand(N, 2) * torch.tensor([W, H])  # (N, 2)  spread across the image
        log_s = torch.log(
            0.02 * max(H, W) * torch.ones(N, 2)
        )  # (N, 2)  small blobs, log space
        theta = torch.zeros(N)  # (N,)    rotation
        color = torch.zeros(N, 3)  # (N, 3)  sigmoid -> 0.5 gray
        op_raw = torch.full((N,), -2.0)  # (N,)    sigmoid -> ~0.12 opacity
        return cls(mu, log_s, theta, color, op_raw)

    def forward(self, H, W, xy=None):
        Sigma = covariance_2d(self.log_s.exp(), self.theta)
        depth_order = torch.arange(self.mu.size(0))
        return render(
            self.mu,
            Sigma,
            self.color.sigmoid(),
            self.op_raw.sigmoid(),
            depth_order,
            H,
            W,
            xy,
        )


class Optimizer2D:
    densify_every = 200  # run a pass every 200 optimization steps
    # grad_threshold = 2e-4  # densify Gaussian i if g_i > grad_threshold
    densify_frac = 0.2  # densify the top % of Gaussians by g_i each pass
    size_threshold = 0.02  # clone if max scale <= 2% of image width, else split
    split_scale = 1.6  # each split child gets (parent scale / split_scale)
    # prune_opacity = 0.005  # remove Gaussian i if its opacity < this
    prune_opacity = 0.02
    chunk_above = 1024  # with more Gaussians than this, render in pixel chunks

    def __init__(self, target: torch.Tensor, budget=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target = target
        self.budget = budget

    def optimize(self, N=None, num_steps=2000, densify=True):
        if N is None:
            N = self.budget // 2
        target = self.target
        H, W = target.size()[:2]

        gaussians = Gaussian2D.random(N, H, W).to(target.device)
        opt = torch.optim.Adam(gaussians.parameters(), lr=1e-2)
        grad_mag = torch.zeros(N, device=target.device)
        t_start = time.perf_counter()
        for step in range(num_steps):
            if densify and step > 0 and step % self.densify_every == 0:
                gaussians = self.densify(gaussians, grad_mag / self.densify_every)
                opt = torch.optim.Adam(gaussians.parameters(), lr=1e-2)
                grad_mag = torch.zeros(gaussians.mu.size(0), device=target.device)
                torch.cuda.empty_cache()

            opt.zero_grad()
            if gaussians.mu.size(0) <= self.chunk_above:
                img = gaussians(H, W)
                loss = ((img - target) ** 2).mean()
                loss.backward()
            else:
                img, loss = self.backward_chunked(gaussians)
            opt.step()
            grad_mag += gaussians.mu.grad.norm(dim=-1)
            # psnr = -10 * torch.log10(loss)

            if step % 100 == 0:
                psnr = -10 * torch.log10(loss)
                print(
                    f"step {step:4d}  loss {loss.item():.5f}  PSNR {psnr.item():.2f} dB  "
                    f"{(time.perf_counter() - t_start) / (step + 1):.4f} sec/step"
                )

        self.final_N = gaussians.mu.size(0)
        return img

    def backward_chunked(self, gaussians: Gaussian2D):
        target = self.target
        H, W = target.size()[:2]
        xy = pixel_grid(H, W).to(target.device)  # (P, 2)
        target_flat = target.reshape(-1, 3)  # (P, 3)
        P_chunk = max(H * W * self.chunk_above // gaussians.mu.size(0), 1)
        img = torch.empty_like(target_flat)
        loss = 0.0
        for p in range(0, H * W, P_chunk):
            out = gaussians(H, W, xy[p : p + P_chunk])  # (P_chunk, 3)
            chunk_loss = ((out - target_flat[p : p + P_chunk]) ** 2).sum()
            chunk_loss = chunk_loss / target.numel()
            chunk_loss.backward()
            img[p : p + P_chunk] = out.detach()
            loss = loss + chunk_loss.detach()
        return img.reshape(H, W, 3), loss

    def densify(self, gaussians: Gaussian2D, grad_mag):
        # grad_mag: per-Gaussian g_i accumulated since the last pass
        # TODO: dense = grad_mag > grad_threshold             (under-fit Gaussians)
        # TODO: clone = dense & (max_scale <= size_threshold)  (duplicate in place)
        # TODO: split = dense & (max_scale >  size_threshold)  (2 children, scale / split_scale)
        # TODO: prune Gaussians with opacity < prune_opacity
        # TODO: keep the total count <= budget
        prune = gaussians.op_raw.sigmoid() < self.prune_opacity
        max_scale = gaussians.log_s.exp().max(dim=-1).values
        W = self.target.size(1)

        # select highest grads to densify
        # dense = (grad_mag > self.grad_threshold) & ~prune
        n_keep = int((~prune).sum())
        room = max(self.budget - n_keep, 0)
        k = min(int(self.densify_frac * grad_mag.size(0)), room, n_keep)
        top = grad_mag.masked_fill(prune, -1).topk(k).indices
        dense = torch.zeros_like(prune)
        dense[top] = True

        clone = dense & (max_scale <= self.size_threshold * W)
        split = dense & (max_scale > self.size_threshold * W)
        keep = ~prune & ~split

        n = torch.arange(grad_mag.size(0), device=grad_mag.device)
        idx = torch.cat([n[keep], n[clone], n[split], n[split]])
        mu, log_s, theta, color, op_raw = (
            p.detach()[idx]
            for p in [
                gaussians.mu,
                gaussians.log_s,
                gaussians.theta,
                gaussians.color,
                gaussians.op_raw,
            ]
        )

        # split children and sample
        c = idx.numel() - 2 * int(split.sum())
        s = log_s[c:].exp()
        cos, sin = theta[c:].cos(), theta[c:].sin()
        R = torch.stack([cos, -sin, sin, cos], dim=-1).view(-1, 2, 2)
        mu[c:] += (R @ (s * torch.randn_like(s))[..., None]).squeeze(-1)
        log_s[c:] -= math.log(self.split_scale)

        print(
            f"densify: {grad_mag.size(0)} -> {idx.numel()}  (clone {int(clone.sum())}, "
            f"split {int(split.sum())}, prune {int(prune.sum())})  "
            f"g_i median {grad_mag.median().item():.2e}  max {grad_mag.max().item():.2e}"
        )
        return Gaussian2D(mu, log_s, theta, color, op_raw)


def quaternion_to_rotation(q):
    # q: (N, 4) as (w, x, y, z)
    # TODO: normalize q, then build R(q) above          -> (N, 3, 3)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.view(-1, 3, 3)


def covariance_3d(scale, quat):
    # scale: (N, 3) positive,  quat: (N, 4)
    # TODO: R = quaternion_to_rotation(quat); return R S S^T R^T  -> (N, 3, 3)
    R = quaternion_to_rotation(quat)
    S = torch.diag_embed(scale)
    Sigma = R @ S @ S.mT @ R.mT
    return Sigma


def project_gaussian(mu3, Sigma3, R_wc, t, K):
    # mu3: (N, 3) world means,  Sigma3: (N, 3, 3) world covariances
    mu_cam = mu3 @ R_wc.T + t  # world -> camera
    # TODO: mu2   = perspective-project mu_cam with K            (N, 2)
    # TODO: J     = Jacobian of the projection at mu_cam         (N, 2, 3)
    # TODO: Scam  = R_wc @ Sigma3 @ R_wc.T                       (N, 3, 3)
    #       Sig2 = J @ Scam @ J.transpose(-1, -2)                (N, 2, 2)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    xc, yc, zc = mu_cam.unbind(-1)
    mu2 = torch.stack([fx * xc / zc + cx, fy * yc / zc + cy], dim=-1)

    zero = torch.zeros_like(zc)
    J = torch.stack(
        [fx / zc, zero, -fx * xc / zc**2, zero, fy / zc, -fy * yc / zc**2],
        dim=-1,
    ).view(-1, 2, 3)

    Scam = R_wc @ Sigma3 @ R_wc.T
    Sig2 = J @ Scam @ J.transpose(-1, -2)

    depth = zc
    return mu2, Sig2, depth


class Camera:
    def __init__(self, path: pathlib.Path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(path, "r") as f:
            cam_json = json.load(f)
        self.H, self.W = cam_json["height"], cam_json["width"]
        self.K = torch.tensor(cam_json["K"], device=device)

        def load(frames):
            return [
                {
                    "img": self.load_image(path.parent / fr["file"]).to(device),
                    "R_wc": torch.tensor(fr["R_wc"], device=device),
                    "t": torch.tensor(fr["t"], device=device),
                }
                for fr in frames
            ]

        self.frames = load(cam_json["frames"])
        self.val_frames = load(cam_json["val_frames"])

    @staticmethod
    def load_image(file):
        img = Image.open(file).convert("RGB")
        return torch.from_numpy(np.array(img)).float() / 255

    def random_frame(self):
        return self.frames[torch.randint(len(self.frames), (1,)).item()]

    def orbit(self, n=16, elev=25.0, tilt=20.0):
        device = self.K.device
        radius = self.frames[0]["t"].norm()
        elev, tilt = math.radians(elev), math.radians(tilt)
        phi = torch.arange(n, device=device) * (2 * math.pi / n)
        ring = torch.stack(
            [
                math.cos(elev) * phi.sin(),
                math.sin(elev) * torch.ones_like(phi),
                math.cos(elev) * phi.cos(),
            ],
            dim=-1,
        )
        Rx = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(tilt), -math.sin(tilt)],
                [0.0, math.sin(tilt), math.cos(tilt)],
            ],
            device=device,
        )
        center = radius * ring @ Rx.T
        forward = -center / radius
        up = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(forward)
        right = torch.linalg.cross(forward, up)
        right = right / right.norm(dim=-1, keepdim=True)
        down = torch.linalg.cross(forward, right)
        R_wc = torch.stack([right, down, forward], dim=1)
        t = -(R_wc @ center[..., None]).squeeze(-1)
        return [{"R_wc": R, "t": tt} for R, tt in zip(R_wc, t)]


class Gaussian3D(nn.Module):

    def __init__(self, mu3, log_s, quat, color, op_raw):
        super().__init__()
        self.mu = nn.Parameter(mu3.detach())
        self.log_s = nn.Parameter(log_s.detach())
        self.quat = nn.Parameter(quat.detach())
        self.color = nn.Parameter(color.detach())
        self.op_raw = nn.Parameter(op_raw.detach())

    @classmethod
    def random(cls, N, H, W) -> Gaussian3D:
        mu3 = (torch.rand(N, 3) * 2 - 1) * 1.5  # (N, 3)  cloud in ~[-1.5, 1.5]^3
        log_s = torch.log(0.08 * torch.ones(N, 3))  # (N, 3)  small 3D blobs
        far = N // 2
        mu3[far:] = (torch.rand(N - far, 3) * 2 - 1) * 50.0
        log_s[far:] = math.log(5.0)
        quat = torch.zeros(N, 4)
        quat[:, 0] = 1.0  # (N, 4)  identity rotation (w, x, y, z)
        color = torch.zeros(N, 3)  # (N, 3)  sigmoid -> gray
        op_raw = torch.full((N,), -2.0)  # (N,)    sigmoid -> low opacity
        return cls(mu3, log_s, quat, color, op_raw)

    def forward(self, camera: Camera, frame, xy=None):
        x, y, z = (self.mu @ frame["R_wc"].T + frame["t"]).unbind(-1)
        vis = (z > 0.2) & (x * x + y * y < (1.5 * z) ** 2)
        Sigma = covariance_3d(self.log_s[vis].exp(), self.quat[vis])
        mu2, sig2, depth = project_gaussian(
            self.mu[vis], Sigma, frame["R_wc"], frame["t"], camera.K
        )
        order = torch.argsort(depth, descending=False)
        return render(
            mu2,
            sig2,
            self.color[vis].sigmoid(),
            self.op_raw[vis].sigmoid(),
            order,
            camera.H,
            camera.W,
            xy,
        )


class Optimizer3D:
    densify_every = 200  # run a pass every 200 optimization steps
    # grad_threshold = 1e-4  # densify Gaussian i if g_i > grad_threshold
    densify_frac = 0.3  # densify the top % of Gaussians by g_i each pass
    size_threshold = 0.15  # clone if max scale <= this, else split
    split_scale = 1.6  # each split child gets (parent scale / split_scale)
    # prune_opacity = 0.005  # remove Gaussian i if its opacity < this
    prune_opacity = 0.12
    chunk_above = 1024

    def __init__(self, camera: Camera, budget=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.camera = camera
        self.budget = budget
        self.gaussians = None

    def optimize(self, N=None, num_steps=2000, densify=True):
        if N is None:
            N = self.budget // 2
        H = self.camera.H
        W = self.camera.W

        gaussians = Gaussian3D.random(N, H, W).to(self.device)
        opt = torch.optim.Adam(gaussians.parameters(), lr=1e-2)
        grad_mag = torch.zeros(N, device=self.device)
        t_start = time.perf_counter()
        for step in range(num_steps):
            if densify and step > 0 and step % self.densify_every == 0:
                gaussians = self.densify(gaussians, grad_mag / self.densify_every)
                opt = torch.optim.Adam(gaussians.parameters(), lr=1e-2)
                grad_mag = torch.zeros(gaussians.mu.size(0), device=self.device)
                torch.cuda.empty_cache()

            frame = self.camera.random_frame()
            target = frame["img"]
            opt.zero_grad()
            if gaussians.mu.size(0) <= self.chunk_above:
                img = gaussians(self.camera, frame)
                loss = ((img - target) ** 2).mean()
                loss.backward()
            else:
                img, loss = self.backward_chunked(gaussians, frame)
            opt.step()
            grad_mag += gaussians.mu.grad.norm(dim=-1)
            # psnr = -10 * torch.log10(loss)

            if step % 100 == 0:
                psnr = -10 * torch.log10(loss)
                print(
                    f"step {step:4d}  loss {loss.item():.5f}  PSNR {psnr.item():.2f} dB  "
                    f"{(time.perf_counter() - t_start) / (step + 1):.4f} sec/step"
                )

        self.final_N = gaussians.mu.size(0)
        self.gaussians = gaussians
        return img

    def backward_chunked(self, gaussians: Gaussian3D, frame):
        target = frame["img"]
        H, W = target.size()[:2]
        xy = pixel_grid(H, W).to(target.device)
        target_flat = target.reshape(-1, 3)
        P_chunk = max(H * W * self.chunk_above // gaussians.mu.size(0), 1)
        img = torch.empty_like(target_flat)
        loss = 0.0
        for p in range(0, H * W, P_chunk):
            out = gaussians(self.camera, frame, xy[p : p + P_chunk])
            chunk_loss = ((out - target_flat[p : p + P_chunk]) ** 2).sum()
            chunk_loss = chunk_loss / target.numel()
            chunk_loss.backward()
            img[p : p + P_chunk] = out.detach()
            loss = loss + chunk_loss.detach()
        return img.reshape(H, W, 3), loss

    def densify(self, gaussians: Gaussian3D, grad_mag):
        # grad_mag: per-Gaussian g_i accumulated since the last pass
        # TODO: dense = grad_mag > grad_threshold             (under-fit Gaussians)
        # TODO: clone = dense & (max_scale <= size_threshold)  (duplicate in place)
        # TODO: split = dense & (max_scale >  size_threshold)  (2 children, scale / split_scale)
        # TODO: prune Gaussians with opacity < prune_opacity
        # TODO: keep the total count <= budget
        prune = gaussians.op_raw.sigmoid() < self.prune_opacity
        max_scale = gaussians.log_s.exp().max(dim=-1).values

        # select highest grads to densify
        # dense = (grad_mag > self.grad_threshold) & ~prune
        # room = max(self.budget - int((~prune).sum()), 0)
        # if dense.sum() > room:
        #     top = grad_mag.masked_fill(~dense, -1).topk(room).indices
        #     dense = torch.zeros_like(dense)
        #     dense[top] = True
        n_keep = int((~prune).sum())
        room = max(self.budget - n_keep, 0)
        k = min(int(self.densify_frac * grad_mag.size(0)), room, n_keep)
        top = grad_mag.masked_fill(prune, -1).topk(k).indices
        dense = torch.zeros_like(prune)
        dense[top] = True

        clone = dense & (max_scale <= self.size_threshold)
        split = dense & (max_scale > self.size_threshold)
        keep = ~prune & ~split

        n = torch.arange(grad_mag.size(0), device=grad_mag.device)
        idx = torch.cat([n[keep], n[clone], n[split], n[split]])
        mu, log_s, quat, color, op_raw = (
            p.detach()[idx]
            for p in [
                gaussians.mu,
                gaussians.log_s,
                gaussians.quat,
                gaussians.color,
                gaussians.op_raw,
            ]
        )

        # split children and sample
        c = idx.numel() - 2 * int(split.sum())
        s = log_s[c:].exp()
        R = quaternion_to_rotation(quat[c:])
        mu[c:] += (R @ (s * torch.randn_like(s))[..., None]).squeeze(-1)
        log_s[c:] -= math.log(self.split_scale)

        print(
            f"densify: {grad_mag.size(0)} -> {idx.numel()}  (clone {int(clone.sum())}, "
            f"split {int(split.sum())}, prune {int(prune.sum())})  "
            f"g_i median {grad_mag.median().item():.2e}  max {grad_mag.max().item():.2e}"
        )
        return Gaussian3D(mu, log_s, quat, color, op_raw)
