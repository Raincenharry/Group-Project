"""Compare PR-ENDO's relit images with VR-Caps ground truth (evaluation plan §9–11).

    python compare.py <VRCaps_scene_vN> <pred_dir> [--calibration eval_relight/<dataset>/calibration.json]
                      [--lpips vgg] [--out-root eval_relight]

For every (config, frame) in lights.json, four images of the same camera are compared:
    G   ground truth, test lighting          G0  ground truth, base lighting (L0)
    P   PR-ENDO, test lighting               B   PR-ENDO, base lighting (L0)
L3 and L5 have two ground-truth versions (noshadow, shadow); P is compared with each.

Output, in eval_relight/<dataset>/<run>/:
    per_image.csv, summary.csv, summary_levels.csv   metrics (plan §11.4–11.5)
    checks.json                                     stage-1 sanity checks on the predictions
    config.json                                     everything needed to reproduce this run
    curves/*.png, panels/<config>/<variant>/<frame>.jpg, panels/_highlights/
"""
import argparse
import csv
import json
import shutil
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

import relight_io as io

NAN = float("nan")
METRICS = ["psnr", "ssim", "lpips", "psnr_base", "ssim_base", "lpips_base", "psnr_gain", "ssim_gain", "lpips_gain",
           "psnr_l0", "psnr_changed", "changed_frac", "ratio_rmse", "ratio_corr", "ratio_bias", "shadow_frac",
           "psnr_aligned", "psnr_scaled", "psnr_base_spotfwd",
           "dir_x_gt", "dir_y_gt", "dir_x_pred", "dir_y_pred", "dir_agree", "n_valid_px"]
INFO = ["scene", "config", "level", "frame", "split", "ref_variant", "intensity_scale", "offset_r", "offset_u",
        "offset_f", "offset_norm", "yaw_deg", "pitch_deg", "beam_angle_deg", "color_r", "color_g", "color_b"]


# ---------- metrics ----------

def psnr(a, b, mask):
    if not mask.any():
        return NAN
    return float(-10 * np.log10(max(float(((a - b) ** 2)[mask].mean()), 1e-12)))


def ssim(a, b, mask):
    """Gaussian-window SSIM (sigma 1.5, 11 px), channel-averaged map, then averaged over the mask."""
    if not mask.any():
        return NAN
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda z: gaussian_filter(z, 1.5, truncate=3.5)
    maps = []
    for c in range(3):
        x, y = a[..., c].astype(np.float64), b[..., c].astype(np.float64)
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        maps.append((2 * mx * my + c1) * (2 * sxy + c2) / ((mx * mx + my * my + c1) * (sxx + syy + c2)))
    return float(np.mean(maps, axis=0)[mask].mean())


class Lpips:
    """LPIPS with pixels outside the mask set to 0 in both images (plan §9.3). Needs `pip install lpips`."""

    def __init__(self, net, max_side):
        import torch
        import lpips
        self.torch, self.max_side = torch, max_side
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()

    def __call__(self, a, b, mask):
        t = self.torch

        def prep(x):
            x = np.where(mask[..., None], x, 0.0).transpose(2, 0, 1)
            ten = t.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None] * 2 - 1
            if self.max_side and max(ten.shape[-2:]) > self.max_side:
                ten = t.nn.functional.interpolate(ten, scale_factor=self.max_side / max(ten.shape[-2:]), mode="area")
            return ten.to(self.device)

        with t.no_grad():
            return float(self.model(prep(a), prep(b)).item())


def log_ratio(num, den):
    return np.log(np.maximum(num, io.EPS)) - np.log(np.maximum(den, io.EPS))


def direction(rho, mask, x, y):
    """Which side of the image the relative brightening leans to: covariance of rho with x (right) and y (up)."""
    r = rho[mask] - rho[mask].mean()
    return float((r * x[mask]).mean()), float((r * y[mask]).mean())


def image_metrics(G_lin, P_lin, G0_lin, B_lin, valid, level, lp, coords):
    G, P, G0, B = (io.encode(z) for z in (G_lin, P_lin, G0_lin, B_lin))
    unsat = np.stack([z.max(-1) for z in (G, P, G0, B)]).max(0) < io.SAT
    lit = (io.luminance(G0_lin) > io.LIT) & (io.luminance(B_lin) > io.LIT)
    changed = valid & (np.abs(G - G0).max(-1) > io.CHANGED)
    m = valid & unsat & lit

    r = {"psnr": psnr(P, G, valid), "ssim": ssim(P, G, valid),
         "psnr_base": psnr(B, G, valid), "ssim_base": ssim(B, G, valid),
         "psnr_l0": psnr(B, G0, valid),
         "psnr_changed": psnr(P, G, changed),
         "changed_frac": float(changed.sum() / max(valid.sum(), 1)),
         "n_valid_px": int(valid.sum())}
    r["psnr_gain"] = r["psnr"] - r["psnr_base"]
    r["ssim_gain"] = r["ssim"] - r["ssim_base"]
    if lp is not None:
        r["lpips"], r["lpips_base"] = lp(P, G, valid), lp(B, G, valid)
        r["lpips_gain"] = r["lpips_base"] - r["lpips"]

    if level in ("L1", "L4"):
        # spatially uniform change: compare per channel, pooled; a correlation has nothing to explain here
        d = (log_ratio(P_lin, B_lin) - log_ratio(G_lin, G0_lin))[m]
        r["ratio_rmse"], r["ratio_bias"] = (float(np.sqrt((d ** 2).mean())), float(d.mean())) if d.size else (NAN, NAN)
        rho_err = (log_ratio(io.luminance(P_lin), io.luminance(B_lin)) - log_ratio(io.luminance(G_lin), io.luminance(G0_lin)))
    else:
        rp = log_ratio(io.luminance(P_lin), io.luminance(B_lin))
        rg = log_ratio(io.luminance(G_lin), io.luminance(G0_lin))
        rho_err = rp - rg
        if m.sum() > 1:
            d = rho_err[m]
            r["ratio_rmse"], r["ratio_bias"] = float(np.sqrt((d ** 2).mean())), float(d.mean())
            sp, sg = rp[m].std(), rg[m].std()
            r["ratio_corr"] = float(np.corrcoef(rp[m], rg[m])[0, 1]) if sp > 1e-9 and sg > 1e-9 else NAN
            if level in ("L2", "L3", "L5"):
                x, y = coords
                r["dir_x_gt"], r["dir_y_gt"] = direction(rg, m, x, y)
                r["dir_x_pred"], r["dir_y_pred"] = direction(rp, m, x, y)
                gx, gy = r["dir_x_gt"], r["dir_y_gt"]
                if max(abs(gx), abs(gy)) > 1e-3:
                    want, got = (gx, r["dir_x_pred"]) if abs(gx) >= abs(gy) else (gy, r["dir_y_pred"])
                    r["dir_agree"] = float(np.sign(want) == np.sign(got))
    return r, {"G": G, "P": P, "G0": G0, "rho_err": np.where(m, rho_err, np.nan)}


# ---------- panels ----------

def _tile(img, size):
    im = Image.fromarray(np.round(np.clip(img, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((size, size), Image.Resampling.BOX)
    return im


def _diverging(v, lim=0.5):
    """Red where the prediction is too bright, blue where too dark, grey where not measured."""
    t = np.clip(np.nan_to_num(v) / lim, -1, 1)[..., None]
    white = np.ones(3)
    rgb = np.where(t > 0, white + t * (np.array([0.85, 0.15, 0.1]) - white),
                   white - t * (np.array([0.1, 0.3, 0.85]) - white))
    return np.where(np.isnan(v)[..., None], 0.5, rgb)


def make_panel(path, imgs, valid, title, size):
    tiles = [imgs["G0"], imgs["G"], imgs["P"],
             np.where(valid[..., None], np.clip(np.abs(imgs["P"] - imgs["G"]) * 4, 0, 1), 0.5),
             _diverging(imgs["rho_err"])]
    tiles = [_tile(t, size) for t in tiles]
    tw, th = tiles[0].size
    panel = Image.new("RGB", (tw * 5 + 16, th + 52), "white")
    draw = ImageDraw.Draw(panel)
    for i, t in enumerate(tiles):
        panel.paste(t, (4 + i * (tw + 2), 48))
    draw.text((6, 4), title[0], fill="black")
    draw.text((6, 18), title[1], fill="black")
    for i, label in enumerate(["G0 (truth, L0)", "G (truth)", "P (PR-ENDO)", "|P-G| x4", "ratio error"]):
        draw.text((6 + i * (tw + 2), 33), label, fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, quality=90)


# ---------- summaries ----------

def nanstat(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (float(np.nanmean(a)), float(np.nanstd(a))) if a.size else (NAN, NAN)


def summarise(rows, keys, thresholds):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in groups.items():
        s = {k: v for k, v in zip(keys, key)}
        for k in INFO:
            if k not in s and k not in ("frame", "split") and len({str(r[k]) for r in rs}) == 1:
                s[k] = rs[0][k]
        s["n_frames"] = len(rs)
        for m in METRICS:
            s[f"{m}_mean"], s[f"{m}_std"] = nanstat(r.get(m) for r in rs)
        if s.get("level") != "L0":
            corr = s["ratio_corr_mean"]
            s["effective"] = bool(s["psnr_gain_mean"] > thresholds["gain_db"] and (np.isnan(corr) or corr > thresholds["corr"]))
            s["trusted"] = bool(s["effective"] and s["psnr_l0_mean"] - s["psnr_mean"] <= thresholds["gap_db"])
        out.append(s)
    return out


def summarise_levels(summary):
    groups = defaultdict(list)
    for s in summary:
        groups[(s["level"], s["ref_variant"])].append(s)
    out = []
    for (level, variant), ss in sorted(groups.items()):
        row = {"level": level, "ref_variant": variant, "n_configs": len(ss), "n_frames": sum(s["n_frames"] for s in ss)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = nanstat(s[f"{m}_mean"] for s in ss)
        out.append(row)
    return out


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})


# ---------- curves ----------

def curves(out, summary, rows, ds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)

    def by(level, variant, xkey):
        pts = defaultdict(list)
        for s in summary:
            if s["level"] == level and s["ref_variant"] == variant and not np.isnan(s.get(xkey, NAN)):
                pts[round(s[xkey], 4)].append(s)
        return sorted(pts.items())

    def line(ax, pts, key, **kw):
        if pts:
            xs = [x for x, _ in pts]
            ax.errorbar(xs, [np.nanmean([s[f"{key}_mean"] for s in ss]) for _, ss in pts],
                        yerr=[np.nanmean([s[f"{key}_std"] for s in ss]) for _, ss in pts], marker="o", ms=4, capsize=2, **kw)

    def save(fig, ax, name, xlabel, ylabel):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)

    l2 = by("L2", "noshadow", "beam_angle_deg")
    if l2:
        fig, ax = plt.subplots(figsize=(6, 4))
        line(ax, l2, "psnr", label="PR-ENDO relit")
        line(ax, l2, "psnr_base", label="no relighting (L0 render)", ls="--")
        line(ax, l2, "psnr_base_spotfwd", label="unturned spot (L2_fwd render)", ls=":")
        line(ax, l2, "psnr_l0", label="reconstruction at L0", color="0.6")
        save(fig, ax, "L2_angle.png", "beam turned away from camera axis (deg)", "PSNR (dB)")
        fig, ax = plt.subplots(figsize=(6, 4))
        line(ax, l2, "ratio_corr", label="PR-ENDO relit")
        save(fig, ax, "L2_ratio.png", "beam turned away from camera axis (deg)", "ratio_corr")

    l3n, l3s = by("L3", "noshadow", "offset_norm"), by("L3", "shadow", "offset_norm")
    if l3n:
        fig, ax = plt.subplots(figsize=(6, 4))
        line(ax, l3n, "psnr", label="PR-ENDO vs truth without shadows")
        line(ax, l3s, "psnr", label="PR-ENDO vs truth with shadows")
        line(ax, l3n, "psnr_base", label="no relighting (L0 render)", ls="--")
        line(ax, l3n, "psnr_l0", label="reconstruction at L0", color="0.6")
        save(fig, ax, "L3_offset.png", "light offset (multiples of d_ref)", "PSNR (dB)")
        fig, ax = plt.subplots(figsize=(6, 4))
        line(ax, l3n, "ratio_corr", label="ratio_corr, no shadows")
        line(ax, l3s, "ratio_corr", label="ratio_corr, with shadows")
        line(ax, l3n, "ratio_rmse", label="ratio_rmse, no shadows", ls="--")
        line(ax, l3s, "ratio_rmse", label="ratio_rmse, with shadows", ls="--")
        save(fig, ax, "L3_ratio.png", "light offset (multiples of d_ref)", "value")

    l5 = [r for r in rows if r["level"] == "L5" and r["split"] == "test"]
    if l5:
        fig, ax = plt.subplots(figsize=(6, 4))
        order = {(e["config"], e["frame"]): i for i, e in enumerate(ds.entries)}
        for i, cfg_id in enumerate(sorted({r["config"] for r in l5})):
            for variant, ls in (("noshadow", "-"), ("shadow", "--")):
                rs = sorted((r for r in l5 if r["config"] == cfg_id and r["ref_variant"] == variant),
                            key=lambda r: order[(r["config"], r["frame"])])
                if rs:
                    ax.plot(range(len(rs)), [r["psnr"] for r in rs], ls, marker="o", ms=3, color=f"C{i}",
                            label=f"{cfg_id} PR-ENDO vs {variant}")
                    if variant == "noshadow":
                        ax.plot(range(len(rs)), [r["psnr_base"] for r in rs], ":", color=f"C{i}", label=f"{cfg_id} no relighting")
        save(fig, ax, "L5_sequence.png", "frame index within segment", "PSNR (dB)")


# ---------- stage-1 checks on the predictions ----------

def prediction_checks(ds, pred, rows, summary, calib, missing):
    def check(status, msg, **kw):
        return {"status": status, "summary": msg, **kw}

    rec = pred.record
    c = {}
    c["predictions_complete"] = check("pass" if not missing else "fail",
                                      "every valid entry has a prediction" if not missing else f"{len(missing)} predictions missing",
                                      missing=missing[:50])
    if rec.get("dataset"):
        c["record_dataset"] = check("pass" if Path(str(rec["dataset"])).name == ds.name else "fail",
                                    f"record.json says {rec['dataset']}, comparing against {ds.name}")
    if calib and rec.get("spot_power") is not None:
        same = abs(float(rec["spot_power"]) - calib["spot_power"]) <= 1e-3 * max(1.0, calib["spot_power"])
        c["spot_power"] = check("pass" if same else "fail", f"record {rec['spot_power']} vs calibration {calib['spot_power']:.4f}")
    calib_l0 = [r["psnr"] for r in rows if r["config"] == ds.l0 and r["split"] == "calib"]
    if calib_l0 and rec.get("train_psnr") is not None:
        gap = float(rec["train_psnr"]) - float(np.mean(calib_l0))
        c["l0_vs_train"] = check("pass" if gap <= 1.0 else "fail",
                                 f"L0 on calibration frames {np.mean(calib_l0):.2f} dB vs training average {rec['train_psnr']} dB")
    for s in summary:
        if s["ref_variant"] != "noshadow":
            continue
        cfg = s["config"]
        if s["level"] in ("L1", "L4"):
            gap = abs(s["psnr_mean"] - s["psnr_scaled_mean"])
            ok = abs(s["ratio_bias_mean"]) <= 0.05 and gap <= 0.5
            c[f"scale_{cfg}"] = check("pass" if ok else "fail",
                                      f"ratio bias {s['ratio_bias_mean']:+.3f}, PSNR {s['psnr_mean']:.2f} vs scaled L0 render {s['psnr_scaled_mean']:.2f}")
        elif s["level"] in ("L2", "L3", "L5") and cfg != "L2_fwd":
            agree = [r["dir_agree"] for r in rows if r["config"] == cfg and r["ref_variant"] == "noshadow"
                     and r["split"] == "test" and not np.isnan(r.get("dir_agree", NAN))]
            if not agree:
                continue
            frac, corr = float(np.mean(agree)), s["ratio_corr_mean"]
            ok = frac >= 0.8 and (np.isnan(corr) or corr > 0)
            c[f"direction_{cfg}"] = check("pass" if ok else "fail",
                                          f"brightening on the same side in {frac:.0%} of {len(agree)} frames, ratio_corr {corr:+.2f}")
    return c


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("pred")
    ap.add_argument("--calibration", help="default: eval_relight/<dataset>/calibration.json if it exists")
    ap.add_argument("--out-root", default="eval_relight")
    ap.add_argument("--run", help="run name, default: date-time")
    ap.add_argument("--lpips", choices=["none", "vgg", "alex"], default="none",
                    help="vgg matches PR-ENDO's own evaluation; needs `pip install lpips` and downloads weights once")
    ap.add_argument("--lpips-max-side", type=int, default=0, help="downscale before LPIPS so the longer side is at most this (0 = full size)")
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--panel-size", type=int, default=320)
    ap.add_argument("--gain-db", type=float, default=1.0, help="'effective' needs mean PSNR gain above this")
    ap.add_argument("--corr", type=float, default=0.8, help="'effective' needs mean ratio_corr above this")
    ap.add_argument("--gap-db", type=float, default=3.0, help="'trusted' needs PSNR within this of the L0 reconstruction")
    args = ap.parse_args()

    ds, pred = io.Dataset(args.dataset), io.Predictions(args.pred)
    scene_dir = Path(args.out_root) / ds.name
    out = scene_dir / (args.run or datetime.now().strftime("%Y%m%d-%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    calib_path = Path(args.calibration) if args.calibration else scene_dir / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else None
    thresholds = {"gain_db": args.gain_db, "corr": args.corr, "gap_db": args.gap_db}
    lp = Lpips(args.lpips, args.lpips_max_side) if args.lpips != "none" else None

    entries = [e for e in ds.entries if e.get("valid", True) and e["config"] in ds.configs]
    by_frame = defaultdict(list)
    for e in entries:
        by_frame[e["frame"]].append(e)

    # Per-channel scale fitted on L0 only, then applied to every config (plan §9.7). Diagnostic, not the main result.
    num, den = np.zeros(3), np.zeros(3)
    for frame in by_frame:
        if ds.split(frame) != "test":
            continue
        g0, b = ds.load_gt(ds.l0, "noshadow", frame), pred.load(ds.l0, frame)
        if g0 is None or b is None:
            continue
        m = (ds.load_depth(frame) > 0) & (io.encode(g0).max(-1) < io.SAT) & (io.encode(b).max(-1) < io.SAT)
        num += (b * g0)[m].sum(0)
        den += (b * b)[m].sum(0)
    align = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)

    rows, missing, panel_of = [], [], {}
    n_frames = len(by_frame)
    for i, (frame, ents) in enumerate(sorted(by_frame.items())):
        print(f"\r[{i + 1}/{n_frames}] frame {frame}", end="", flush=True)
        G0_lin, B_lin = ds.load_gt(ds.l0, "noshadow", frame), pred.load(ds.l0, frame)
        if G0_lin is None or B_lin is None:
            missing += [f"{e['config']}/{frame} (no L0 for this frame)" for e in ents]
            continue
        valid = ds.load_depth(frame) > 0
        h, w = valid.shape
        x, y = io.pixel_grid(ds.relight_meta, w, h)
        coords = (x / np.abs(x).max(), y / np.abs(y).max())
        P_fwd = pred.load("L2_fwd", frame) if any(ds.configs[e["config"]].get("level") == "L2" for e in ents) else None

        for e in ents:
            cfg = ds.configs[e["config"]]
            P_lin = pred.load(cfg["id"], frame)
            if P_lin is None:
                missing.append(f"{cfg['id']}/{frame}")
                continue
            gts = {v: ds.load_gt(cfg["id"], v, frame) for v in ds.variants(cfg["id"])}
            gts = {v: g for v, g in gts.items() if g is not None}
            shadow_frac = NAN
            if "noshadow" in gts and "shadow" in gts:
                shadowed = valid & (io.luminance(gts["noshadow"]) > 1.1 * io.luminance(gts["shadow"]))
                shadow_frac = float(shadowed.sum() / max(valid.sum(), 1))
            o = cfg.get("offset_dref") or [NAN, NAN, NAN]
            col = cfg.get("color_scale") or [1.0, 1.0, 1.0]
            for variant, G_lin in gts.items():
                row = {"scene": ds.name, "config": cfg["id"], "level": cfg.get("level"), "frame": frame,
                       "split": ds.split(frame), "ref_variant": variant,
                       "intensity_scale": cfg.get("intensity_scale", 1.0),
                       "offset_r": o[0], "offset_u": o[1], "offset_f": o[2],
                       "offset_norm": float(np.linalg.norm(o)) if cfg.get("offset_dref") else NAN,
                       "yaw_deg": cfg.get("yaw_deg", 0.0), "pitch_deg": cfg.get("pitch_deg", 0.0),
                       "beam_angle_deg": io.beam_angle_deg(cfg),
                       "color_r": col[0], "color_g": col[1], "color_b": col[2]}
                row.update({m: NAN for m in METRICS})
                met, imgs = image_metrics(G_lin, P_lin, G0_lin, B_lin, valid, cfg.get("level"), lp, coords)
                row.update(met)
                row["shadow_frac"] = shadow_frac
                G = io.encode(G_lin)
                row["psnr_aligned"] = psnr(io.encode(P_lin * align), G, valid)
                if cfg.get("level") in ("L1", "L4"):
                    row["psnr_scaled"] = psnr(io.encode(B_lin * cfg.get("intensity_scale", 1.0) * np.asarray(col)), G, valid)
                if cfg.get("level") == "L2" and cfg["id"] != "L2_fwd" and P_fwd is not None:
                    row["psnr_base_spotfwd"] = psnr(io.encode(P_fwd), G, valid)
                rows.append(row)
                if not args.no_panels:
                    p = out / "panels" / cfg["id"] / variant / f"{frame}.jpg"
                    title = (f"{cfg['id']}   frame {frame}   truth: {variant}   ({row['split']})",
                             f"PSNR {row['psnr']:.2f} dB   gain {row['psnr_gain']:+.2f} dB   "
                             f"ratio_corr {row['ratio_corr']:+.2f}   ratio_rmse {row['ratio_rmse']:.3f}")
                    make_panel(p, imgs, valid, title, args.panel_size)
                    panel_of[(cfg["id"], variant, frame)] = p
    print()

    test_rows = [r for r in rows if r["split"] == "test"]
    summary = summarise(test_rows, ["config", "ref_variant"], thresholds)
    summary.sort(key=lambda s: (s["level"], s["config"], s["ref_variant"]))
    levels = summarise_levels(summary)
    write_csv(out / "per_image.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "summary_levels.csv", levels)

    if panel_of:
        hl = out / "panels" / "_highlights"
        hl.mkdir(parents=True, exist_ok=True)
        groups = defaultdict(list)
        for r in test_rows:
            if not np.isnan(r["psnr_gain"]):
                groups[(r["config"], r["ref_variant"])].append(r)
        for (cfg_id, variant), rs in groups.items():
            rs.sort(key=lambda r: r["psnr_gain"])
            for tag, r in (("worst", rs[0]), ("median", rs[len(rs) // 2]), ("best", rs[-1])):
                shutil.copy(panel_of[(cfg_id, variant, r["frame"])], hl / f"{cfg_id}__{variant}__{tag}_{r['frame']}.jpg")

    try:
        curves(out / "curves", summary, rows, ds)
    except ImportError:
        print("matplotlib not available, curves skipped")

    checks = prediction_checks(ds, pred, rows, summary, calib, missing)
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=float))
    if calib:
        shutil.copy(calib_path, out / "calibration.json")
    (out / "config.json").write_text(json.dumps({
        "dataset": str(Path(args.dataset).resolve()), "pred": str(Path(args.pred).resolve()),
        "run": out.name, "created": datetime.now().isoformat(timespec="seconds"),
        "record": pred.record, "calibration": str(calib_path) if calib else None,
        "spot_power": calib["spot_power"] if calib else None,
        "thresholds": thresholds,
        "masks": {"saturated_display": io.SAT, "lit_linear_luminance": io.LIT, "changed_display": io.CHANGED,
                  "log_floor": io.EPS, "shadow_ratio": 1.1},
        "align_scale_rgb": align.tolist(),
        "lpips": {"net": args.lpips, "max_side": args.lpips_max_side},
        "exr_backend": io.exr_backend(),
        "export_settings": ds.export,
    }, indent=2, default=float))

    print(f"{len(rows)} comparisons, {len(missing)} missing predictions")
    for name, c in checks.items():
        print(f"[{c['status'].upper():4}] {name:24} {c['summary']}")
    print("\nper level (mean over configs, truth without shadows):")
    for r in levels:
        if r["ref_variant"] == "noshadow":
            corr = "n/a" if np.isnan(r["ratio_corr_mean"]) else f"{r['ratio_corr_mean']:+.2f}"
            print(f"  {r['level']}: PSNR {r['psnr_mean']:.2f}  gain {r['psnr_gain_mean']:+.2f}  "
                  f"ratio_corr {corr}  ratio_rmse {r['ratio_rmse_mean']:.3f}  ({r['n_configs']} configs)")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
