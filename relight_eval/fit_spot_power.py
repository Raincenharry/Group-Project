"""Fit PR-ENDO's spotlight falloff exponent n to the Unity spotlight (evaluation plan §8.2).

    python fit_spot_power.py <VRCaps_scene_vN> [--out-root eval_relight]

On each calibration frame, the forward-facing spot (L2_fwd) and the point light (L0) sit at the
same place with the same intensity, so their per-pixel luminance ratio depends only on the angle
alpha between the pixel's ray and the beam axis. PR-ENDO models that ratio as cos(alpha)^n; we pick
the n that fits Unity's measured ratio best in least squares. Only ground-truth images are used.

Writes eval_relight/<dataset>/calibration.json and spot_fit.png. The n in it is what goes to the
PR-ENDO teammate; once sent, it is not refitted.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import relight_io as io

N_MAX = 1000.0


def collect(ds, fwd_id, frames):
    alphas, ratios, used = [], [], []
    sat = io.SAT ** io.GAMMA
    for frame in frames:
        g0 = ds.load_gt(ds.l0, "noshadow", frame)
        gs = ds.load_gt(fwd_id, "noshadow", frame)
        if g0 is None or gs is None:
            continue
        h, w = g0.shape[:2]
        y0, ys = io.luminance(g0), io.luminance(gs)
        m = (ds.load_depth(frame) > 0) & (y0 > 0.02) & (g0.max(-1) < sat) & (gs.max(-1) < sat)
        alphas.append(io.angle_to_axis(ds.relight_meta, w, h)[m])
        ratios.append(ys[m] / y0[m])
        used.append(frame)
    if not used:
        raise SystemExit(f"no calibration frame has both {ds.l0} and {fwd_id}")
    return np.concatenate(alphas), np.concatenate(ratios), used


def fit(alpha, ratio):
    """Least-squares n for ratio ≈ cos(alpha)^n; returns n and the RMS residual."""
    cos = np.cos(alpha)
    loss = lambda log_n: float(np.mean((cos ** np.exp(log_n) - ratio) ** 2))
    best = minimize_scalar(loss, bounds=(np.log(1e-3), np.log(N_MAX)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(best.x)), float(np.sqrt(best.fun))


def on_axis(alpha, ratio, min_pixels=200):
    """Median ratio over the pixels closest to the beam axis, and the angle that took in.

    Inside the inner cone a Unity spot given the point light's intensity is exactly as bright as it,
    so this should be 1. The view centre is often the dark far lumen and masked out, hence the
    widening instead of a fixed angle.
    """
    deg = np.sort(np.degrees(alpha))
    if deg.size < min_pixels:
        return float("nan"), float("nan")
    limit = float(deg[min_pixels - 1])
    return float(np.median(ratio[np.degrees(alpha) <= limit])), limit


def profile(alpha, ratio, step_deg=1.0):
    deg = np.degrees(alpha)
    rows = []
    for lo in np.arange(0, deg.max() + step_deg, step_deg):
        m = (deg >= lo) & (deg < lo + step_deg)
        if m.sum() >= 20:
            rows.append([round(lo + step_deg / 2, 2), float(np.median(ratio[m])), int(m.sum())])
    return rows


def plot(path, prof, n, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = np.array([r[0] for r in prof])
    a = np.radians(np.linspace(0, deg.max(), 200))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(deg, [r[1] for r in prof], "o", ms=3, color="0.3", label="Unity: L2_fwd / L0 (median per 1°)")
    ax.plot(np.degrees(a), np.cos(a) ** n, color="C0", label=f"cos^n, n = {n:.2f} (used)")
    for key in ("spot_inner_deg", "spot_outer_deg"):
        if cfg.get(key):
            ax.axvline(cfg[key], color="0.7", lw=0.8, ls=":")
    ax.set(xlabel="angle from beam axis (deg)", ylabel="relative brightness", ylim=(0, 1.15))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--fwd", default="L2_fwd", help="config id of the forward-facing spot")
    ap.add_argument("--split", default="calib", help="frames whose split is this are used")
    ap.add_argument("--out-root", default="eval_relight")
    args = ap.parse_args()

    ds = io.Dataset(args.dataset)
    frames = sorted(f for f in ds.relight_frames if ds.split(f) == args.split)
    alpha, ratio, used = collect(ds, args.fwd, frames)
    n, rmse = fit(alpha, ratio)
    axis_ratio, axis_deg = on_axis(alpha, ratio)
    prof = profile(alpha, ratio)
    cfg = ds.configs[args.fwd]
    inner = cfg.get("spot_inner_deg")
    axis_checkable = inner is not None and axis_deg <= inner

    out_dir = Path(args.out_root) / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = {
        "dataset": str(ds.root),
        "fitted": datetime.now().isoformat(timespec="seconds"),
        "spot_power": n,
        "spot_fit_rmse": rmse,
        "on_axis_ratio": axis_ratio,
        "on_axis_within_deg": axis_deg,
        "frames": used,
        "n_pixels": int(alpha.size),
        "spot_inner_deg": cfg.get("spot_inner_deg"),
        "spot_outer_deg": cfg.get("spot_outer_deg"),
        "spot_profile": prof,
    }
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    plot(out_dir / "spot_fit.png", prof, n, cfg)

    print(f"n = {n:.3f}   rmse = {rmse:.4f}   ({alpha.size} pixels from {len(used)} frames)")
    print(f"near-axis brightness, spot / point: {axis_ratio:.3f} (pixels within {axis_deg:.1f} deg of the axis)")
    if not axis_checkable:
        print(f"  not checked: needs pixels inside the inner cone ({inner} deg); see spot_fit.png")
    elif not abs(axis_ratio - 1) <= 0.05:
        print("warning: inside the inner cone the Unity spot should be exactly as bright as the point light; it is "
              f"{axis_ratio - 1:+.1%} off, and every L2 comparison will carry that bias. Check the spot's intensity settings.")
    print(f"written {out_dir / 'calibration.json'} and spot_fit.png")


if __name__ == "__main__":
    main()
