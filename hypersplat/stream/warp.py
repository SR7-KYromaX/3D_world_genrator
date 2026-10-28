"""Depth-image-based novel views: a real keyframe re-projected to a nearby pseudo camera.

Online Difix used to start every pseudo view from a render of the live splats, which early in
a run (or on a low-parallax capture) is mostly fog that Difix can only guess around. A real
photo with its Depth Anything map (aligned to ACE, hypersplat/stream/depth.py) and its ACE
pose already holds most of what a camera a few degrees away sees:

    source pixels --depth, K--> 3D points --c2w_src, then the pseudo camera's w2c--> its pixels

Points are forward-splatted with a z-buffer (the nearest surface wins) as 2x2 dots, so the
small magnification of an orbit leaves no cracks. What the source never saw (disocclusions,
the border the camera moved into) stays invalid; the caller fills it from the splat render
and Difix cleans the seams and the holes.

OpenCV cameras (+Z forward, +Y down), z-depth, any frame as long as depth, poses and K agree.
"""

from typing import Tuple

import cv2
import numpy as np
import torch

# 2x2 dot: the primary pixel plus right / down / diagonal copies a hair further away, so a
# point's own pixel always beats a neighbour's spill-over at equal depth
_OFFSETS = ((0, 0, 1.0), (1, 0, 1.0 + 1e-4), (0, 1, 1.0 + 1e-4), (1, 1, 1.0 + 2e-4))


@torch.no_grad()
def warp_to_view(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, c2w_src: np.ndarray,
                 c2w_dst: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Re-project ``rgb`` (H, W, 3 uint8) with its z-depth (any resolution, 0 = invalid) from
    the camera ``c2w_src`` into ``c2w_dst`` (same K, same size). Returns (image uint8, valid)."""
    H, W = rgb.shape[:2]
    d = depth.astype(np.float32)
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    dev = torch.device(device)
    z = torch.from_numpy(d).to(dev).reshape(-1)
    keep = torch.isfinite(z) & (z > 0)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
                          torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    u, v, z = u.reshape(-1)[keep], v.reshape(-1)[keep], z[keep]
    cols = torch.from_numpy(np.ascontiguousarray(rgb)).to(dev).reshape(-1, 3)[keep]
    pts = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)

    rel = torch.from_numpy((np.linalg.inv(c2w_dst) @ c2w_src).astype(np.float32)).to(dev)
    pts = pts @ rel[:3, :3].T + rel[:3, 3]
    zd = pts[:, 2]
    front = zd > 1e-6
    pts, zd, cols = pts[front], zd[front], cols[front]
    px = torch.floor(fx * pts[:, 0] / zd + cx).long()
    py = torch.floor(fy * pts[:, 1] / zd + cy).long()

    pix, zs, cs = [], [], []
    for dx, dy, bias in _OFFSETS:
        x, y = px + dx, py + dy
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        pix.append(y[ok] * W + x[ok])
        zs.append(zd[ok] * bias)
        cs.append(cols[ok])
    pix, zs, cs = torch.cat(pix), torch.cat(zs), torch.cat(cs)

    zbuf = torch.full((H * W,), float("inf"), device=dev)
    zbuf.scatter_reduce_(0, pix, zs, reduce="amin")
    win = zs <= zbuf[pix]
    out = torch.zeros((H * W, 3), dtype=torch.uint8, device=dev)
    out[pix[win]] = cs[win]
    valid = torch.isfinite(zbuf)
    return out.reshape(H, W, 3).cpu().numpy(), valid.reshape(H, W).cpu().numpy()


def composite(warped: np.ndarray, valid: np.ndarray, render: np.ndarray) -> np.ndarray:
    """Real re-projected pixels where they landed, the splat render in the holes."""
    if render.shape != warped.shape:
        render = cv2.resize(render, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.where(valid[..., None], warped, render)
