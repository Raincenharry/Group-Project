"""Shared helpers for the relighting comparison: image IO, colour encoding, camera geometry,
and the two folder layouts we receive.

Dataset layout: handoff/对接说明_VRCaps组员.md §3–6.
Prediction layout: handoff/对接说明_PRENDO组员.md §4.4.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # must be set before cv2 is imported
try:
    import OpenEXR
except ImportError:
    OpenEXR = None
if os.environ.get("RELIGHT_EXR_BACKEND") == "cv2":
    OpenEXR = None
try:
    import cv2
except ImportError:
    cv2 = None

GAMMA = 2.2
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SAT = 0.98      # display-space value treated as saturated (mask M_unsat)
LIT = 0.01      # linear luminance below which a pixel is too dark for ratios (mask M_lit)
CHANGED = 0.05  # display-space difference that counts as "lighting changed here" (mask M_changed)
EPS = LIT       # floor before log ratios: truth dark where the prediction is lit still counts, but stays bounded


def encode(lin):
    """E(x): clamp to [0, 1], then raise to 1/2.2. Same encoding as the training PNGs and PR-ENDO's output."""
    return np.clip(lin, 0.0, 1.0) ** (1.0 / GAMMA)


def decode(disp):
    return np.clip(disp, 0.0, 1.0) ** GAMMA


def luminance(rgb):
    return rgb @ LUMA


def exr_backend():
    return "openexr" if OpenEXR is not None else ("cv2" if cv2 is not None else "none")


# ---------- image IO ----------

def _openexr_channels(path):
    with OpenEXR.File(str(path), separate_channels=True) as f:
        return {name: np.asarray(ch.pixels, dtype=np.float32) for name, ch in f.channels().items()}


def _cv2_read(path):
    if cv2 is None:
        raise RuntimeError("reading EXR needs the OpenEXR package or opencv-python")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img.astype(np.float32)


def read_exr_rgb(path):
    """Linear RGB, HxWx3 float32. Takes R, G, B with no prefix, else the layer whose name has 'Combined', else the first layer."""
    if OpenEXR is None:
        img = _cv2_read(path)
        return np.repeat(img[..., None], 3, axis=-1) if img.ndim == 2 else img[..., [2, 1, 0]]
    chans = _openexr_channels(path)
    layers = sorted({n.rsplit(".", 1)[0] + "." for n in chans if "." in n},
                    key=lambda p: ("combined" not in p.lower(), p))
    for prefix in [""] + layers:
        names = [prefix + c for c in "RGB"]
        if all(n in chans for n in names):
            return np.stack([chans[n] for n in names], axis=-1)
    raise ValueError(f"{path}: no R, G, B channels among {sorted(chans)}")


def read_exr_depth(path, channel=None):
    """Single-channel depth. Uses `channel` when given and present, else guesses (see the VR-Caps handoff §5.3)."""
    if OpenEXR is None:
        img = _cv2_read(path)
        return img if img.ndim == 2 else img[..., 2]  # cv2 stores BGR: index 2 is R
    chans = _openexr_channels(path)
    if channel and channel in chans:
        return chans[channel]
    if len(chans) == 1:
        return next(iter(chans.values()))
    for name in chans:
        if name == "Z" or name.endswith(".Z") or "depth" in name.lower():
            return chans[name]
    for name in ("R", "Y"):
        if name in chans:
            return chans[name]
    raise ValueError(f"{path}: cannot tell which of {sorted(chans)} is depth; set depth_channel in export_settings.json")


def write_exr(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if OpenEXR is not None:
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        with OpenEXR.File(header, {"Z": arr} if arr.ndim == 2 else {"RGB": arr}) as f:
            f.write(str(path))
    else:
        cv2.imwrite(str(path), arr if arr.ndim == 2 else arr[..., [2, 1, 0]])


def read_png(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def write_png(path, disp):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(disp, 0, 1) * 255).astype(np.uint8)).save(path)


# ---------- cameras ----------

def frame_id(x):
    """Frame numbers are unpadded in transforms json and 4-digit in file names."""
    return str(x).zfill(4)


def load_transforms(path):
    meta = json.loads(Path(path).read_text())
    return meta, {frame_id(f["file_path"]): f for f in meta["frames"]}


def intrinsics(meta, w, h):
    fx = meta.get("fl_x", 0.5 * w / np.tan(0.5 * meta["camera_angle_x"]))
    return fx, meta.get("fl_y", fx), meta.get("cx", w / 2), meta.get("cy", h / 2)


def camera_axes(c2w):
    """Right, up, forward unit vectors and centre of an OpenGL camera-to-world matrix."""
    m = np.asarray(c2w, dtype=np.float64)
    return m[:3, 0], m[:3, 1], -m[:3, 2], m[:3, 3]


def pixel_grid(meta, w, h):
    """Per-pixel ray offsets (x right, y up) on the z = 1 plane, for pixel centres."""
    fx, fy, cx, cy = intrinsics(meta, w, h)
    x = (np.arange(w) + 0.5 - cx) / fx
    y = -(np.arange(h) + 0.5 - cy) / fy
    return np.broadcast_to(x[None, :], (h, w)), np.broadcast_to(y[:, None], (h, w))


def angle_to_axis(meta, w, h):
    """Angle in radians between each pixel's ray and the optical axis."""
    x, y = pixel_grid(meta, w, h)
    return np.arctan(np.hypot(x, y))


def expected_light(cfg, c2w, d_ref):
    """Light position and beam direction implied by a config's relative parameters (VR-Caps handoff §6.2)."""
    r, u, f, c = camera_axes(c2w)
    o = cfg.get("offset_dref") or [0.0, 0.0, 0.0]
    pos = c + d_ref * (o[0] * r + o[1] * u + o[2] * f)
    psi, theta = np.radians(cfg.get("yaw_deg") or 0.0), np.radians(cfg.get("pitch_deg") or 0.0)
    d = np.cos(theta) * (np.cos(psi) * f + np.sin(psi) * r) + np.sin(theta) * u
    return pos, d


def beam_angle_deg(cfg):
    """How far a spot's beam is turned away from the camera axis; NaN for point lights."""
    if cfg.get("type") != "spot":
        return float("nan")
    psi, theta = np.radians(cfg.get("yaw_deg") or 0.0), np.radians(cfg.get("pitch_deg") or 0.0)
    return float(np.degrees(np.arccos(np.clip(np.cos(psi) * np.cos(theta), -1, 1))))


# ---------- the two folders we receive ----------

class Dataset:
    """VRCaps_<scene>_v<version>/ from the VR-Caps teammate."""

    def __init__(self, root):
        self.root = Path(root)
        self.relight = self.root / "relight"
        self.name = self.root.name
        self.lights = json.loads((self.relight / "lights.json").read_text())
        self.relight_meta, self.relight_frames = load_transforms(self.relight / "transforms_relight.json")
        tpath = self.root / "transforms.json"
        self.train_meta, self.train_frames = load_transforms(tpath) if tpath.exists() else ({}, {})
        epath = self.relight / "export_settings.json"
        self.export = json.loads(epath.read_text()) if epath.exists() else {}
        self.configs = {c["id"]: c for c in self.lights["configs"]}
        self.entries = [dict(e, frame=frame_id(e["frame"])) for e in self.lights["frames"]]
        self.d_ref = float(self.lights["d_ref"])
        self.l0 = next(c["id"] for c in self.lights["configs"] if c.get("level") == "L0")

    def split(self, frame):
        return self.relight_frames[frame].get("split", "test")

    def c2w(self, frame):
        return self.relight_frames[frame]["transform_matrix"]

    def variants(self, cfg_id):
        return self.configs[cfg_id].get("variants") or ["noshadow"]

    def gt_path(self, cfg_id, variant, frame):
        return self.relight / "gt" / cfg_id / variant / f"{frame}.exr"

    def load_gt(self, cfg_id, variant, frame):
        p = self.gt_path(cfg_id, variant, frame)
        return read_exr_rgb(p) if p.exists() else None

    def load_depth(self, frame):
        return read_exr_depth(self.relight / "depth" / f"{frame}.exr", self.export.get("depth_channel"))


class Predictions:
    """pred_<dataset>_<run>/ from the PR-ENDO teammate."""

    def __init__(self, root):
        self.root = Path(root)
        rpath = self.root / "record.json"
        self.record = json.loads(rpath.read_text()) if rpath.exists() else {}

    def load(self, cfg_id, frame):
        """Linear RGB; prefers the EXR (not quantised), falls back to decoding the PNG."""
        exr = self.root / cfg_id / f"{frame}.exr"
        if exr.exists():
            return read_exr_rgb(exr)
        png = self.root / cfg_id / f"{frame}.png"
        return decode(read_png(png)) if png.exists() else None
