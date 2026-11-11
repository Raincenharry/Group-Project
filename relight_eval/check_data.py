"""Check a dataset from the VR-Caps teammate before using it (VR-Caps handoff §8).

    python check_data.py <VRCaps_scene_vN> [--out-root eval_relight]

Prints one line per check and writes eval_relight/<dataset>/check_data.json.
Exit code 1 if any check fails, so the result can go straight into the receipt sent back.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import relight_io as io


def result(status, summary, **details):
    return {"status": status, "summary": summary, **details}


def check_files(ds):
    missing = []
    for name in ("transforms.json", "relight/export_settings.json"):
        if not (ds.root / name).exists():
            missing.append(name)
    for e in ds.entries:
        if not e.get("valid", True):
            continue
        for v in ds.variants(e["config"]):
            if not ds.gt_path(e["config"], v, e["frame"]).exists():
                missing.append(str(ds.gt_path(e["config"], v, e["frame"]).relative_to(ds.root)))
    for frame in ds.relight_frames:
        p = ds.relight / "depth" / f"{frame}.exr"
        if not p.exists():
            missing.append(str(p.relative_to(ds.root)))
    unknown = sorted({e["config"] for e in ds.entries} - set(ds.configs))
    unlisted = sorted({e["frame"] for e in ds.entries} - set(ds.relight_frames))
    ok = not (missing or unknown or unlisted)
    return result("pass" if ok else "fail",
                  "all files present" if ok else f"{len(missing)} missing files, {len(unknown)} unknown configs, {len(unlisted)} frames not in transforms_relight.json",
                  missing=missing[:50], unknown_configs=unknown, frames_not_listed=unlisted)


def check_frames_consistent(ds):
    if not ds.train_frames:
        return result("skip", "transforms.json missing")
    bad = []
    for frame, f in ds.relight_frames.items():
        t = ds.train_frames.get(frame)
        if t is None:
            bad.append(f"{frame}: not a training frame")
        elif not np.allclose(t["transform_matrix"], f["transform_matrix"], atol=1e-6):
            bad.append(f"{frame}: camera matrix differs from transforms.json")
    for key in ("camera_angle_x", "fl_x", "fl_y", "cx", "cy"):
        if ds.train_meta.get(key) != ds.relight_meta.get(key):
            bad.append(f"{key} differs: {ds.train_meta.get(key)} vs {ds.relight_meta.get(key)}")
    return result("pass" if not bad else "fail", "relight frames match the training frames" if not bad else f"{len(bad)} mismatches", problems=bad[:50])


def check_light_pose(ds):
    bad, fixed = [], {}
    for e in ds.entries:
        cfg = ds.configs.get(e["config"])
        if cfg is None or e["frame"] not in ds.relight_frames:
            continue
        pos, d = np.asarray(e["light_pos_world"], float), np.asarray(e["light_dir_world"], float)
        tag = f"{e['config']}/{e['frame']}"
        if abs(np.linalg.norm(d) - 1) > 1e-3:
            bad.append(f"{tag}: light_dir_world is not a unit vector")
        if cfg.get("world_fixed"):
            fixed.setdefault(e["config"], []).append((e["frame"], pos))
            continue
        exp_pos, exp_d = io.expected_light(cfg, ds.c2w(e["frame"]), ds.d_ref)
        err = np.linalg.norm(pos - exp_pos) / ds.d_ref
        if err > 1e-3:
            bad.append(f"{tag}: light position off by {err:.4f} d_ref from what offset_dref implies")
        if cfg.get("type") == "spot":
            ang = np.degrees(np.arccos(np.clip(np.dot(d / np.linalg.norm(d), exp_d), -1, 1)))
            if ang > 0.5:
                bad.append(f"{tag}: beam direction off by {ang:.2f} deg from what yaw/pitch imply")
    for cfg_id, items in fixed.items():
        first_frame, first_pos = items[0]
        _, _, _, centre = io.camera_axes(ds.c2w(first_frame))
        if np.linalg.norm(first_pos - centre) / ds.d_ref > 1e-3:
            bad.append(f"{cfg_id}: fixed light is not at the camera centre of its first frame {first_frame}")
        for frame, pos in items[1:]:
            if np.linalg.norm(pos - first_pos) / ds.d_ref > 1e-6:
                bad.append(f"{cfg_id}/{frame}: fixed light moved")
    return result("pass" if not bad else "fail", "light poses match the config parameters" if not bad else f"{len(bad)} problems", problems=bad[:50])


def check_depth(ds, max_frames):
    """d_ref should be the median training depth; the relight depth should be the training depth of the same frame."""
    problems, depths = [], []
    train_dir = ds.root / "depth_train"
    frames = sorted(ds.train_frames)
    for frame in frames[:: max(1, len(frames) // max_frames)]:
        p = train_dir / f"{frame}.exr"
        if p.exists():
            d = io.read_exr_depth(p, ds.export.get("depth_channel"))
            depths.append(d[d > 0])
    if not depths:
        return result("skip", "no training depth found")
    median = float(np.median(np.concatenate(depths)))
    if abs(median - ds.d_ref) / median > 0.05:
        problems.append(f"d_ref {ds.d_ref:.4g} vs median training depth {median:.4g}")
    for frame in ds.relight_frames:
        a, b = ds.relight / "depth" / f"{frame}.exr", train_dir / f"{frame}.exr"
        if a.exists() and b.exists():
            da, db = ds.load_depth(frame), io.read_exr_depth(b, ds.export.get("depth_channel"))
            if da.shape != db.shape or not np.allclose(da, db, rtol=1e-4, atol=1e-6):
                problems.append(f"{frame}: relight/depth differs from depth_train")
    return result("pass" if not problems else "fail", f"median training depth {median:.4g}, d_ref {ds.d_ref:.4g}",
                  median_train_depth=median, problems=problems[:50])


def _ratio_stats(num, den, mask):
    r = num[mask] / den[mask]
    return float(np.median(r)), r


def check_scaling(ds, level, expected_of, name):
    """Linear export: an intensity or colour change must scale every pixel by exactly that factor."""
    rows, failed = [], False
    for e in ds.entries:
        cfg = ds.configs.get(e["config"])
        if cfg is None or cfg.get("level") != level or not e.get("valid", True):
            continue
        g = ds.load_gt(e["config"], "noshadow", e["frame"])
        g0 = ds.load_gt(ds.l0, "noshadow", e["frame"])
        if g is None or g0 is None:
            continue
        valid = ds.load_depth(e["frame"]) > 0
        sat = io.SAT ** io.GAMMA
        for c, want in enumerate(expected_of(cfg)):
            m = valid & (g0[..., c] > io.LIT) & (g0.max(-1) < sat) & (g.max(-1) < sat)
            if m.sum() < 100:
                continue
            med, r = _ratio_stats(g[..., c], g0[..., c], m)
            p99 = float(np.percentile(np.abs(r / want - 1), 99))
            ok = abs(med / want - 1) <= 0.01 and p99 <= 0.01
            failed |= not ok
            rows.append({"config": e["config"], "frame": e["frame"], "channel": "RGB"[c], "expected": want,
                         "median": round(med, 5), "p99_rel_err": round(p99, 5), "ok": ok})
    if not rows:
        return result("skip", f"no {level} images to check")
    worst = max(rows, key=lambda r: r["p99_rel_err"])
    return result("fail" if failed else "pass",
                  f"{name}: worst 99th-percentile error {worst['p99_rel_err']:.2%} ({worst['config']}/{worst['frame']} {worst['channel']})",
                  rows=rows)


def check_encoding(ds):
    """L0 of a relight frame is the same render as its training frame, so E(L0) must equal the training PNG."""
    rows = []
    for frame in ds.relight_frames:
        png = ds.root / "train_views" / f"{frame}.png"
        g0 = ds.load_gt(ds.l0, "noshadow", frame)
        if g0 is None or not png.exists():
            continue
        diff = np.abs(np.round(io.encode(g0) * 255) - np.round(io.read_png(png) * 255))
        rows.append({"frame": frame, "max_diff_255": int(diff.max()), "frac_over_1": float((diff > 1).mean())})
    if not rows:
        return result("skip", "no L0 image with a matching training PNG")
    worst = max(rows, key=lambda r: r["max_diff_255"])
    return result("pass" if worst["max_diff_255"] <= 1 else "fail",
                  f"largest difference {worst['max_diff_255']}/255 at frame {worst['frame']}", rows=rows)


def check_overexposure(ds, max_frames):
    pngs = sorted((ds.root / "train_views").glob("*.png"))
    if not pngs:
        return result("skip", "no training PNGs")
    fracs = [float((io.read_png(p).max(-1) >= 1.0).mean()) for p in pngs[:: max(1, len(pngs) // max_frames)]]
    mean = float(np.mean(fracs))
    return result("pass" if mean < 0.01 else "fail", f"{mean:.2%} of training pixels at 255 (limit 1%)", worst=max(fracs))


def check_shadow(ds):
    rows, failed = [], False
    for e in ds.entries:
        if not e.get("valid", True) or set(ds.variants(e["config"])) < {"noshadow", "shadow"}:
            continue
        ns = ds.load_gt(e["config"], "noshadow", e["frame"])
        sh = ds.load_gt(e["config"], "shadow", e["frame"])
        if ns is None or sh is None:
            continue
        valid = ds.load_depth(e["frame"]) > 0
        brighter = valid & (io.luminance(sh) > io.luminance(ns) * 1.01 + 1e-4)
        shadowed = valid & (io.luminance(ns) > io.luminance(sh) * 1.1)
        frac = float(brighter.sum() / max(valid.sum(), 1))
        failed |= frac > 1e-4
        rows.append({"config": e["config"], "frame": e["frame"], "frac_shadow_brighter": frac,
                     "shadow_frac": float(shadowed.sum() / max(valid.sum(), 1))})
    if not rows:
        return result("skip", "no config with both shadow and noshadow")
    worst = max(rows, key=lambda r: r["frac_shadow_brighter"])
    return result("fail" if failed else "pass",
                  f"shadow brighter than noshadow on at most {worst['frac_shadow_brighter']:.3%} of pixels; "
                  f"mean shadow area {np.mean([r['shadow_frac'] for r in rows]):.1%}", rows=rows)


def check_export_settings(ds):
    if not ds.export:
        return result("fail", "export_settings.json missing or empty")
    return result("info", f"{len(ds.export)} fields recorded: {', '.join(sorted(ds.export))}", settings=ds.export)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out-root", default="eval_relight")
    ap.add_argument("--max-frames", type=int, default=30, help="training frames sampled for depth and exposure checks")
    args = ap.parse_args()

    ds = io.Dataset(args.dataset)
    checks = {
        "files": check_files(ds),
        "frames_consistent": check_frames_consistent(ds),
        "light_pose": check_light_pose(ds),
        "depth": check_depth(ds, args.max_frames),
        "linear_intensity": check_scaling(ds, "L1", lambda c: [c["intensity_scale"]], "intensity"),
        "colour_scale": check_scaling(ds, "L4", lambda c: c["color_scale"], "colour"),
        "encoding": check_encoding(ds),
        "overexposure": check_overexposure(ds, args.max_frames),
        "shadow": check_shadow(ds),
        "export_settings": check_export_settings(ds),
    }
    for name, c in checks.items():
        print(f"[{c['status'].upper():4}] {name:18} {c['summary']}")

    out = Path(args.out_root) / ds.name / "check_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": str(ds.root), "checked": datetime.now().isoformat(timespec="seconds"),
                               "exr_backend": io.exr_backend(), "checks": checks}, indent=2, default=float))
    print(f"\nwritten {out}")
    sys.exit(1 if any(c["status"] == "fail" for c in checks.values()) else 0)


if __name__ == "__main__":
    main()
