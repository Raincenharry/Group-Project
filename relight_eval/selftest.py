"""End-to-end self-test on a synthetic tube scene. Needs no real data and no GPU.

    python selftest.py [--workdir DIR]

Builds a small dataset in the exact layout of the VR-Caps handoff (ground truth rendered with a
Unity-like smoothstep spotlight), runs check_data.py and fit_spot_power.py on it, renders
"PR-ENDO" predictions with the PR-ENDO light model (cos^n spot, slightly wrong albedo), then runs
compare.py. Also checks that the checks catch three planted mistakes:
  - ground truth exported non-linearly          -> check_data.py linear_intensity fails
  - predictions with yaw and offsets mirrored    -> compare.py direction checks fail
  - predictions made with the wrong n            -> compare.py spot_power check fails
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import relight_io as io

HERE = Path(__file__).resolve().parent
W, H, FOV_X = 96, 80, 1.2
RADIUS = 1.0
N_TRAIN = 48
TEST, CALIB, SEGMENTS = [6, 18, 30, 42], [12, 36], [[21, 23, 25], [33, 35, 37]]
SPOT_INNER, SPOT_OUTER = 5.0, 30.0


def configs():
    out = [{"id": "L0", "level": "L0", "type": "point"}]
    out += [{"id": f"L1_s{s}", "level": "L1", "type": "point", "intensity_scale": s} for s in (0.5, 2.0)]
    out += [{"id": "L2_fwd", "level": "L2", "type": "spot"}]
    out += [{"id": f"L2_yaw{a:+d}", "level": "L2", "type": "spot", "yaw_deg": a} for a in (10, -10, 20, -20, 40, -40)]
    out += [{"id": f"L2_pitch{a:+d}", "level": "L2", "type": "spot", "pitch_deg": a} for a in (20, -20)]
    out += [{"id": f"L3_r{v:.2f}", "level": "L3", "type": "point", "offset_dref": [v, 0, 0]} for v in (0.10, 0.25, 0.50, 1.00)]
    out += [{"id": f"L3_u{v:.2f}", "level": "L3", "type": "point", "offset_dref": [0, v, 0]} for v in (0.25, 0.50)]
    out += [{"id": f"L3_f{v:+.2f}", "level": "L3", "type": "point", "offset_dref": [0, 0, v]} for v in (0.25, -0.25)]
    out += [{"id": f"L4_{n}", "level": "L4", "type": "point", "color_scale": c}
            for n, c in (("red", [1, .5, .5]), ("green", [.5, 1, .5]), ("blue", [.5, .5, 1]))]
    out += [{"id": f"L5_seg{i + 1}", "level": "L5", "type": "point", "world_fixed": True} for i in range(len(SEGMENTS))]
    for c in out:
        c.setdefault("intensity_scale", 1.0)
        c.setdefault("color_scale", [1, 1, 1])
        c.setdefault("offset_dref", None if c.get("world_fixed") else [0, 0, 0])
        c.setdefault("yaw_deg", 0)
        c.setdefault("pitch_deg", 0)
        c.setdefault("world_fixed", False)
        spot = c["type"] == "spot"
        c["spot_inner_deg"], c["spot_outer_deg"] = (SPOT_INNER, SPOT_OUTER) if spot else (None, None)
        c["variants"] = ["noshadow", "shadow"] if c["level"] in ("L3", "L5") else ["noshadow"]
    return out


def c2w(k):
    yaw, pitch = 0.1 * np.sin(0.2 * k), 0.05 * np.cos(0.17 * k)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    m[:3, 3] = [0.2 * np.sin(0.3 * k), 0.15 * np.cos(0.23 * k), -0.1 * k]
    return m


def geometry(m):
    """Camera inside the tube x²+y² = R², looking down -z. Returns hit points, inward normals, albedo, z-depth."""
    meta = {"camera_angle_x": FOV_X}
    x, y = io.pixel_grid(meta, W, H)
    d = np.stack([x, y, -np.ones_like(x)], -1) @ m[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    c = m[:3, 3]
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (c[0] * d[..., 0] + c[1] * d[..., 1])
    cc = c[0] ** 2 + c[1] ** 2 - RADIUS ** 2
    t = (-b + np.sqrt(np.maximum(b * b - 4 * a * cc, 0))) / np.maximum(2 * a, 1e-12)
    valid = (a > 1e-8) & (t < 15)
    X = c + t[..., None] * d
    n = -np.stack([X[..., 0], X[..., 1], np.zeros_like(t)], -1) / RADIUS
    theta = np.arctan2(X[..., 1], X[..., 0])
    tex = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(7 * theta) * np.cos(4 * X[..., 2]))
    albedo = tex[..., None] * np.array([0.8, 0.4, 0.35])
    depth = np.where(valid, t * (d @ -m[:3, 2]), 0.0)
    return {"X": X, "n": n, "albedo": albedo, "theta": theta, "valid": valid, "depth": depth}


def shade(g, pos, beam, spot, intensity, colour, albedo=None):
    L = pos - g["X"]
    dist = np.linalg.norm(L, axis=-1)
    l = L / dist[..., None]
    cos_i = np.clip((g["n"] * l).sum(-1), 0, None)
    s = np.ones_like(dist) if spot is None else spot(np.clip((-l * beam).sum(-1), -1, 1))
    a = g["albedo"] if albedo is None else albedo
    rgb = intensity * a * (cos_i * s / dist ** 2 / np.pi)[..., None] * np.asarray(colour)
    return np.where(g["valid"][..., None], rgb, 0.0).astype(np.float32)


def unity_spot(cos_a):
    lo, hi = np.cos(np.radians(SPOT_OUTER)), np.cos(np.radians(SPOT_INNER))
    t = np.clip((cos_a - lo) / (hi - lo), 0, 1)
    return t * t


def light_for(cfg, frame_k, seg_first, d_ref, mirror=False):
    m = c2w(frame_k)
    if cfg["world_fixed"]:
        first = c2w(seg_first)
        return first[:3, 3].copy(), -first[:3, 2]
    c = dict(cfg)
    if mirror:  # a sign error on the prediction side
        c["yaw_deg"] = -c["yaw_deg"]
        c["offset_dref"] = [-c["offset_dref"][0], -c["offset_dref"][1], c["offset_dref"][2]]
    return io.expected_light(c, m, d_ref)


def build_dataset(root):
    cfgs = configs()
    geoms = {k: geometry(c2w(k)) for k in range(N_TRAIN)}
    d_ref = float(np.median(np.concatenate([g["depth"][g["valid"]] for g in geoms.values()])))
    probe = shade(geoms[0], c2w(0)[:3, 3], None, None, 1.0, [1, 1, 1])
    i0 = float(0.8 / np.percentile(probe.max(-1)[geoms[0]["valid"]], 99.5))

    meta = {"camera_angle_x": FOV_X}
    frames = [{"file_path": str(k), "transform_matrix": c2w(k).tolist()} for k in range(N_TRAIN)]
    (root / "train_views").mkdir(parents=True, exist_ok=True)
    (root / "transforms.json").write_text(json.dumps(dict(meta, frames=frames), indent=1))
    for k, g in geoms.items():
        f = io.frame_id(k)
        io.write_png(root / "train_views" / f"{f}.png", io.encode(shade(g, c2w(k)[:3, 3], None, None, i0, [1, 1, 1])))
        io.write_exr(root / "depth_train" / f"{f}.exr", g["depth"].astype(np.float32))

    seq = [k for s in SEGMENTS for k in s]
    rel = sorted(set(TEST + CALIB + seq))
    split = {k: ("calib" if k in CALIB else "test") for k in rel}
    rel_frames = [{"file_path": str(k), "split": split[k], "transform_matrix": c2w(k).tolist()} for k in rel]
    (root / "relight").mkdir(exist_ok=True)
    (root / "relight" / "transforms_relight.json").write_text(json.dumps(dict(meta, frames=rel_frames), indent=1))
    for k in rel:
        io.write_exr(root / "relight" / "depth" / f"{io.frame_id(k)}.exr", geoms[k]["depth"].astype(np.float32))

    plan = []
    for c in cfgs:
        if c["level"] == "L5":
            seg = SEGMENTS[int(c["id"][-1]) - 1]
            plan += [(c, k, seg[0]) for k in seg]
        else:
            ks = TEST + (seq if c["id"] == "L0" else [])
            ks += CALIB if c["id"] in ("L0", "L2_fwd", "L2_yaw+20", "L3_r0.25") else []
            plan += [(c, k, None) for k in ks]
    entries = []
    for c, k, first in plan:
        pos, beam = light_for(c, k, first, d_ref)
        ok = bool(np.hypot(pos[0], pos[1]) < RADIUS - 0.05 * d_ref)
        entries.append({"config": c["id"], "frame": io.frame_id(k), "light_pos_world": pos.tolist(),
                        "light_dir_world": beam.tolist(), "valid": ok})
        if not ok:
            continue
        img = shade(geoms[k], pos, beam, unity_spot if c["type"] == "spot" else None,
                    i0 * c["intensity_scale"], c["color_scale"])
        for v in c["variants"]:  # a tube seen from inside casts no shadows, so both versions are the same render
            io.write_exr(root / "relight" / "gt" / c["id"] / v / f"{io.frame_id(k)}.exr", img)

    lights = {"mm_per_unit": None, "d_ref": d_ref, "base_light": {"type": "point", "intensity": i0, "color": [1, 1, 1]},
              "configs": cfgs, "frames": entries}
    (root / "relight" / "lights.json").write_text(json.dumps(lights, indent=1))
    (root / "relight" / "export_settings.json").write_text(json.dumps({
        "renderer": "selftest.py synthetic tube", "color_space": "linear", "tone_mapping": "off",
        "light_falloff": "inverse square", "shadows": "none (convex scene)", "depth_channel": "Z"}, indent=1))
    return {"geoms": geoms, "d_ref": d_ref, "i0": i0, "cfgs": {c["id"]: c for c in cfgs}, "entries": entries}


def build_predictions(root, scene, dataset_name, n, mirror=False):
    seg_first = {f"L5_seg{i + 1}": s[0] for i, s in enumerate(SEGMENTS)}
    psnrs = []
    for e in scene["entries"]:
        if not e["valid"]:
            continue
        c, k = scene["cfgs"][e["config"]], int(e["frame"])
        g = scene["geoms"][k]
        # reconstruction error: fixed per frame, like a trained model's, so relighting a frame cannot change it
        noise = np.random.default_rng(k).normal(0, 0.02, g["theta"].shape)
        albedo = g["albedo"] * (1 + 0.04 * np.sin(13 * g["theta"] + 2 * g["X"][..., 2]) + noise)[..., None]
        pos, beam = light_for(c, k, seg_first.get(c["id"]), scene["d_ref"], mirror)
        spot = (lambda ca: np.where(ca > 0, np.abs(ca) ** n, 0.0)) if c["type"] == "spot" else None
        lin = shade(g, pos, beam, spot, scene["i0"] * c["intensity_scale"], [1, 1, 1], albedo)
        lin = (io.encode(lin) ** io.GAMMA * np.asarray(c["color_scale"])).astype(np.float32)  # colour applied outside, as in the handoff
        io.write_exr(root / c["id"] / f"{e['frame']}.exr", lin)
        io.write_png(root / c["id"] / f"{e['frame']}.png", io.encode(lin))
        if c["id"] == "L0":
            gt = shade(g, c2w(k)[:3, 3], None, None, scene["i0"], [1, 1, 1])
            psnrs.append(-10 * np.log10(((io.encode(lin) - io.encode(gt)) ** 2)[g["valid"]].mean()))
    (root / "record.json").write_text(json.dumps({
        "dataset": dataset_name, "prendo_commit": "synthetic", "code_commit": "synthetic",
        "train_command": "none", "iteration": 30000, "depth_scale": 1.0, "spot_power": n,
        "train_psnr": float(np.mean(psnrs)), "train_ssim": None, "train_lpips": None,
        "notes": "mirrored light offsets" if mirror else ""}, indent=1))


def run(script, *args):
    p = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], capture_output=True, text=True, cwd=HERE)
    return p.returncode, p.stdout + p.stderr


def latest_run(out_root, name):
    runs = sorted(p for p in (out_root / name).iterdir() if p.is_dir())
    return runs[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir")
    ap.add_argument("--keep", action="store_true", help="keep the workdir")
    args = ap.parse_args()
    work = Path(args.workdir or tempfile.mkdtemp(prefix="relight_selftest_"))
    out_root = work / "eval_relight"
    results = []

    def expect(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

    ds_dir = work / "VRCaps_synth_v1"
    print(f"workdir {work}  (EXR backend: {io.exr_backend()})\nbuilding synthetic dataset …")
    scene = build_dataset(ds_dir)

    code, log = run("check_data.py", ds_dir, "--out-root", out_root)
    expect("check_data passes on a correct dataset", code == 0, "" if code == 0 else log[-1500:])

    bad = work / "VRCaps_synth_v1_nonlinear"
    shutil.copytree(ds_dir, bad)
    for p in (bad / "relight" / "gt" / "L1_s2.0" / "noshadow").glob("*.exr"):
        io.write_exr(p, io.read_exr_rgb(p) ** 0.9)
    code, log = run("check_data.py", bad, "--out-root", out_root)
    rep = json.loads((out_root / bad.name / "check_data.json").read_text())["checks"]
    expect("check_data catches non-linear export", code == 1 and rep["linear_intensity"]["status"] == "fail",
           rep["linear_intensity"]["summary"])

    code, log = run("fit_spot_power.py", ds_dir, "--out-root", out_root)
    calib = json.loads((out_root / ds_dir.name / "calibration.json").read_text())
    n = calib["spot_power"]
    expect("fit_spot_power runs", code == 0 and np.isfinite(n), f"n = {n:.2f}, rmse {calib['spot_fit_rmse']:.3f}")
    if calib["on_axis_within_deg"] <= calib["spot_inner_deg"]:
        expect("near-axis spot brightness equals the point light", abs(calib["on_axis_ratio"] - 1) < 0.02,
               f"{calib['on_axis_ratio']:.3f} within {calib['on_axis_within_deg']:.1f} deg")
    else:
        expect("near-axis check reported as not checkable", "not checked" in log,
               f"nearest usable pixels {calib['on_axis_within_deg']:.1f} deg off axis, inner cone {calib['spot_inner_deg']} deg")

    pred_ok = work / "pred_ok"
    build_predictions(pred_ok, scene, ds_dir.name, n)
    code, log = run("compare.py", ds_dir, pred_ok, "--out-root", out_root, "--run", "ok")
    expect("compare runs on correct predictions", code == 0, "" if code == 0 else log[-2000:])
    if code == 0:
        res = out_root / ds_dir.name / "ok"
        checks = json.loads((res / "checks.json").read_text())
        failed = [k for k, v in checks.items() if v["status"] == "fail"]
        expect("all prediction checks pass", not failed, ", ".join(failed))
        summary = {(r["config"], r["ref_variant"]): r for r in _read_csv(res / "summary.csv")}
        l3 = summary.get(("L3_r0.50", "noshadow"), {})
        expect("relighting beats no relighting on L3_r0.50", float(l3.get("psnr_gain_mean") or 0) > 1,
               f"gain {float(l3.get('psnr_gain_mean') or 0):+.2f} dB")
        l1 = summary.get(("L1_s0.5", "noshadow"), {})
        expect("L1 prediction equals the scaled L0 render", abs(float(l1["psnr_mean"]) - float(l1["psnr_scaled_mean"])) < 0.5,
               f"{float(l1['psnr_mean']):.2f} vs {float(l1['psnr_scaled_mean']):.2f} dB")
        expect("invalid L3_r1.00 entries skipped", ("L3_r1.00", "noshadow") not in summary)
        expect("curves and panels written", (res / "curves" / "L3_offset.png").exists()
               and any((res / "panels" / "_highlights").iterdir()))

    pred_bad = work / "pred_mirrored"
    build_predictions(pred_bad, scene, ds_dir.name, n, mirror=True)
    code, log = run("compare.py", ds_dir, pred_bad, "--out-root", out_root, "--run", "mirrored", "--no-panels")
    checks = json.loads((out_root / ds_dir.name / "mirrored" / "checks.json").read_text())
    caught = [k for k in ("direction_L2_yaw+20", "direction_L3_r0.25") if checks.get(k, {}).get("status") == "fail"]
    expect("compare catches mirrored yaw and offsets", len(caught) == 2, ", ".join(caught))

    pred_n = work / "pred_wrong_n"
    build_predictions(pred_n, scene, ds_dir.name, n * 2)
    code, log = run("compare.py", ds_dir, pred_n, "--out-root", out_root, "--run", "wrong_n", "--no-panels")
    checks = json.loads((out_root / ds_dir.name / "wrong_n" / "checks.json").read_text())
    expect("compare catches predictions made with the wrong n", checks.get("spot_power", {}).get("status") == "fail")

    ok = all(r for _, r in results)
    print(f"\n{sum(r for _, r in results)}/{len(results)} passed")
    if args.keep or not ok:
        print(f"outputs kept in {work}")
    else:
        shutil.rmtree(work)
    sys.exit(0 if ok else 1)


def _read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
