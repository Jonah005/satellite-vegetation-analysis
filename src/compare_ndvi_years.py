# compare_ndvi_years.py — OG arrays everywhere: 3×3 grid + hotspots + JSON-constrained LLM + deterministic fallback
from pathlib import Path
import os, sys, io, json, base64, re, collections, math
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).parent.resolve()
IN_2018_DIR = ROOT / "2018"
IN_2024_DIR = ROOT / "2024"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Tiny vision model; low RAM
MODEL = "moondream"
OLLAMA_URL_CHAT = "http://127.0.0.1:11434/api/chat"
GEN_OPTIONS = {"num_ctx": 512, "num_predict": 220, "temperature": 0.2}

# Fixed, shared display ranges (do not change for each image)
NDVI_VMIN, NDVI_VMAX = -0.2, 0.8
DIFF_VMIN, DIFF_VMAX = -0.4, 0.4
LOSS_THR, GAIN_THR = -0.15, 0.15

# For hotspot selection
K_HOTSPOTS = 5
BANNED_VISUALS = ["tree", "shadow", "road", "building", "car", "person", "cloud"]

# ---------------- I/O helpers ----------------
def find_ndvi(folder: Path) -> Path:
    for pat in ("*NDVI*.tif", "*NDVI*.tiff", "*ndvi*.tif", "*ndvi*.tiff", "NDVI.tif", "NDVI.tiff"):
        hits = sorted(folder.glob(pat))
        if hits: return hits[0]
    raise FileNotFoundError(f"No NDVI GeoTIFF in {folder}")

def read_band1(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        meta = src.meta.copy()
    nodata = meta.get("nodata", None)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    # Keep OG NDVI; only drop impossible outliers
    arr = np.where((arr < -5) | (arr > 5), np.nan, arr)
    return arr, meta

def align_to_ref(src_arr, src_meta, ref_meta):
    same = (
        src_meta["crs"] == ref_meta["crs"] and
        src_meta["transform"] == ref_meta["transform"] and
        (src_meta["width"], src_meta["height"]) == (ref_meta["width"], ref_meta["height"])
    )
    if same: return src_arr
    dst = np.empty((ref_meta["height"], ref_meta["width"]), dtype="float32")
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_meta["transform"], src_crs=src_meta["crs"],
        dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
        resampling=Resampling.bilinear,
    )
    dst[~np.isfinite(dst)] = np.nan
    return dst

# ---------------- stats ----------------
def nanstats(arr):
    a = arr[np.isfinite(arr)]
    if a.size == 0: return {"count": 0}
    q = np.nanpercentile(a, [5,25,50,75,95]).tolist()
    return {
        "count": int(a.size),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "min": float(np.nanmin(a)),
        "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
        "max": float(np.nanmax(a)),
    }

def sector_means_3x3(arr):
    """Mean NDVI per 3×3 sector using labels NW,N,NE,W,C,E,SW,S,SE."""
    H, W = arr.shape
    rows = np.linspace(0, H, 4, dtype=int)
    cols = np.linspace(0, W, 4, dtype=int)
    labels = ["NW","N","NE","W","C","E","SW","S","SE"]
    out, k = {}, 0
    for i in range(3):
        for j in range(3):
            r0, r1 = rows[i], rows[i+1]
            c0, c1 = cols[j], cols[j+1]
            m = float(np.nanmean(arr[r0:r1, c0:c1]))
            out[labels[k]] = (None if np.isnan(m) else m); k += 1
    return out

def grid_stats(arr18, arr24, diff, grid=(3,3)):
    """Per-cell means and gain/loss counts on a 3×3 (or m×n) grid."""
    H, W = arr18.shape
    gh, gw = grid
    rows = np.linspace(0, H, gh+1, dtype=int)
    cols = np.linspace(0, W, gw+1, dtype=int)
    labels = ["NW","N","NE","W","C","E","SW","S","SE"]; lab_idx = 0

    cells = []
    for i in range(gh):
        for j in range(gw):
            r0, r1 = rows[i], rows[i+1]
            c0, c1 = cols[j], cols[j+1]
            a18 = arr18[r0:r1, c0:c1]
            a24 = arr24[r0:r1, c0:c1]
            d   = diff[r0:r1, c0:c1]

            m18 = float(np.nanmean(a18))
            m24 = float(np.nanmean(a24))
            md  = float(np.nanmean(d))
            gpx = int(((d >= GAIN_THR) & np.isfinite(d)).sum())
            lpx = int(((d <= LOSS_THR) & np.isfinite(d)).sum())

            cid = labels[lab_idx] if grid==(3,3) else f"r{i+1}c{j+1}"; lab_idx += (1 if grid==(3,3) else 0)
            cells.append({
                "id": cid,
                "m18": None if np.isnan(m18) else m18,
                "m24": None if np.isnan(m24) else m24,
                "d":   None if np.isnan(md)  else md,
                "gain_px": gpx,
                "loss_px": lpx,
                "bbox": [int(r0), int(r1), int(c0), int(c1)]
            })
    return {"shape": [gh, gw], "cells": cells}

def hist8(arr, vmin, vmax, bins=10):
    a = arr[np.isfinite(arr)]
    if a.size == 0: return {"bins": [], "counts": []}
    counts, edges = np.histogram(a, bins=bins, range=(vmin, vmax))
    return {"bins": edges.round(3).tolist(), "counts": counts.tolist()}

def change_buckets(diff_arr):
    a = diff_arr[np.isfinite(diff_arr)]
    if a.size == 0: return {}
    return {
        "loss_small":  int((a <= -0.05).sum()),
        "loss_med":    int((a <= -0.15).sum()),
        "loss_strong": int((a <= -0.30).sum()),
        "gain_small":  int((a >=  0.05).sum()),
        "gain_med":    int((a >=  0.15).sum()),
        "gain_strong": int((a >=  0.30).sum()),
    }

# ---------------- hotspots ----------------
def label_components(mask: np.ndarray):
    H, W = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r in range(H):
        if not mask[r].any(): continue
        for c in np.nonzero(mask[r])[0]:
            if visited[r, c]: continue
            q = collections.deque([(r, c)])
            visited[r, c] = True
            pixels = []
            while q:
                rr, cc = q.popleft()
                pixels.append((rr, cc))
                for nr, nc in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True; q.append((nr, nc))
            area = len(pixels)
            if area == 0: continue
            rows = np.fromiter((p[0] for p in pixels), dtype=np.int32, count=area)
            cols = np.fromiter((p[1] for p in pixels), dtype=np.int32, count=area)
            comps.append((area, float(rows.mean()), float(cols.mean()), pixels))
    return comps

def sector_id_of_point_3x3(r, c, H, W):
    ri = 0 if r < H/3 else (1 if r < 2*H/3 else 2)
    ci = 0 if c < W/3 else (1 if c < 2*W/3 else 2)
    labels = [["NW","N","NE"], ["W","C","E"], ["SW","S","SE"]]
    return labels[ri][ci]

def extract_hotspots(diff, kind="gain", k=K_HOTSPOTS):
    H, W = diff.shape
    mask = ((diff >= GAIN_THR) if kind=="gain" else (diff <= LOSS_THR)) & np.isfinite(diff)
    comps = label_components(mask)
    comps.sort(key=lambda x: x[0], reverse=True)
    top = comps[:k]
    out = []
    for area, cr, cc, pixels in top:
        vals = [diff[r, c] for (r, c) in pixels if np.isfinite(diff[r, c])]
        dmean = float(np.mean(vals)) if vals else 0.0
        out.append({
            "type": kind,
            "area_px": int(area),
            "delta_mean": dmean,
            "centroid_rc": [float(cr), float(cc)],
            # keep key name "quadrant" for back-compat; value is 3×3 sector id
            "quadrant": sector_id_of_point_3x3(cr, cc, H, W)
        })
    return out

# ---------------- viz (NO per-image rescale; use a single shared Normalize) ----------------
SHARED_NORM = Normalize(vmin=NDVI_VMIN, vmax=NDVI_VMAX, clip=True)
DIFF_NORM   = Normalize(vmin=DIFF_VMIN, vmax=DIFF_VMAX, clip=True)

def side_by_side_png(arrA, arrB, cmap, out_path: Path, labels=("2018 NDVI","2024 NDVI")):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    im0 = axes[0].imshow(arrA, norm=SHARED_NORM, cmap=cmap, interpolation="nearest")
    axes[0].set_title(labels[0]); axes[0].axis("off")
    im1 = axes[1].imshow(arrB, norm=SHARED_NORM, cmap=cmap, interpolation="nearest")
    axes[1].set_title(labels[1]); axes[1].axis("off")
    fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)

def to_png(arr, cmap, out_path: Path, title=None, diff=False):
    plt.figure(figsize=(6,6))
    norm = DIFF_NORM if diff else SHARED_NORM
    im = plt.imshow(arr, norm=norm, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title: plt.title(title, fontsize=10)
    plt.axis("off"); plt.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close()

def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")

# ---------------- LLM (JSON constrained) ----------------
def build_prompt(facts_json: dict) -> str:
    return (
        "You are an environmental analyst comparing two NDVI maps (2018 left, 2024 right). "
        "Use ONLY the numeric facts below; do not guess objects.\n\n"
        f"Facts: {json.dumps(facts_json, separators=(',',':'))}\n\n"
        "Return a STRICT JSON object:\n"
        "{\n"
        '  "trend": "greening|browning|mixed",\n'
        '  "means": {"y2018": number, "y2024": number, "diff": number},\n'
        '  "spatial": "where change concentrates using 3x3 grid ids (NW,N,NE,W,C,E,SW,S,SE) and hotspot sectors",\n'
        '  "scale": "compare gain_pixels vs loss_pixels (no %)",\n'
        '  "causes": "NDVI-based explanation (e.g., canopy recovery, cropping intensity, moisture season). No trees/roads/clouds/people.",\n'
        '  "next_steps": "two concrete checks (e.g., verify strongest grid cells; sample more dates)."\n'
        "}\n"
        "Rules: use the given numbers; keep concise; no mentions of visual objects."
    )

def chat_vision_json(model: str, prompt: str, images_b64: list[str], timeout_s=240) -> dict | None:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt, "images": images_b64}], "stream": False, "options": GEN_OPTIONS}
    try:
        r = requests.post(OLLAMA_URL_CHAT, json=payload, timeout=timeout_s)
    except requests.exceptions.ConnectionError:
        return None
    if r.status_code != 200: return None
    data = r.json()
    text = (data.get("message") or {}).get("content") or data.get("content") or ""
    if not text: return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def violates_banned(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in BANNED_VISUALS)

# ---------------- deterministic formatter ----------------
def deterministic_report(facts: dict) -> str:
    m18 = facts["ndvi_mean_2018"]; m24 = facts["ndvi_mean_2024"]; md = facts["ndvi_mean_diff"]
    gains = facts["gain_pixels"]; losses = facts["loss_pixels"]
    best_cell = max(facts["grid"]["cells"], key=lambda c: (c["d"] if c["d"] is not None else -1e9))
    strongest_label = best_cell["id"]
    lines = []
    lines.append(f"Overall, NDVI increased from {m18:.3f} (2018) to {m24:.3f} (2024), a mean change of {md:.3f}.")
    lines.append(f"Gains substantially exceed losses (gain pixels: {gains:,}; loss pixels: {losses:,}).")
    lines.append(f"Spatially, the strongest positive change appears in the {strongest_label} grid cell; adjacent cells with positive deltas indicate a broader cluster.")
    lines.append("Interpretation: likely canopy recovery or higher cropping intensity; seasonal moisture differences may contribute.")
    lines.append("Next steps: field-verify the strongest grid cells and inspect additional dates to separate seasonal vs structural change.")
    return "\n".join(lines)

# ---------------- main ----------------
def main():
    p2018 = find_ndvi(IN_2018_DIR)
    p2024 = find_ndvi(IN_2024_DIR)
    ndvi18, meta18 = read_band1(p2018)
    ndvi24, meta24 = read_band1(p2024)
    ndvi24a = align_to_ref(ndvi24, meta24, meta18)  # alignment only; values kept OG

    # Diff computed from OG arrays (no clipping)
    diff = (ndvi24a - ndvi18).astype("float32")
    diff[~np.isfinite(ndvi18) | ~np.isfinite(ndvi24a)] = np.nan

    # Render images with shared norm; no per-panel scaling, no smoothing
    side_png = OUT_DIR / "ndvi_2018_2024_side_by_side.png"
    diff_png = OUT_DIR / "ndvi_diff_2018_2024.png"
    side_by_side_png(ndvi18, ndvi24a, "RdYlGn", side_png)
    to_png(diff, "bwr", diff_png, title="NDVI change (2024 - 2018)", diff=True)

    # Global facts (from OG arrays)
    loss_mask = (diff <= LOSS_THR) & np.isfinite(diff)
    gain_mask = (diff >= GAIN_THR) & np.isfinite(diff)

    grid = grid_stats(ndvi18, ndvi24a, diff, grid=(3,3))
    gain_hot = extract_hotspots(diff, "gain", k=K_HOTSPOTS)
    loss_hot = extract_hotspots(diff, "loss", k=K_HOTSPOTS)

    facts = {
        "image_shape": {"height": int(ndvi18.shape[0]), "width": int(ndvi18.shape[1])},
        "valid_pixels": {"2018": int(np.isfinite(ndvi18).sum()),
                         "2024": int(np.isfinite(ndvi24a).sum()),
                         "diff":  int(np.isfinite(diff).sum())},
        "ndvi_2018_stats": nanstats(ndvi18),
        "ndvi_2024_stats": nanstats(ndvi24a),
        "ndvi_diff_stats":  nanstats(diff),
        "grid_means_3x3": {"2018": sector_means_3x3(ndvi18),
                           "2024": sector_means_3x3(ndvi24a),
                           "diff":  sector_means_3x3(diff)},
        "grid": grid,
        "hotspots": {"gain_top": gain_hot, "loss_top": loss_hot},
        "histograms": {"ndvi_2018": hist8(ndvi18, NDVI_VMIN, NDVI_VMAX, 10),
                       "ndvi_2024": hist8(ndvi24a, NDVI_VMIN, NDVI_VMAX, 10),
                       "ndvi_diff": hist8(diff, DIFF_VMIN, DIFF_VMAX, 12)},
        "thresholds": {"loss_threshold": LOSS_THR, "gain_threshold": GAIN_THR,
                       "bucket_counts": change_buckets(diff)},
        "ndvi_mean_2018": float(np.nanmean(ndvi18)),
        "ndvi_mean_2024": float(np.nanmean(ndvi24a)),
        "ndvi_mean_diff": float(np.nanmean(diff)),
        "loss_pixels": int(loss_mask.sum()),
        "gain_pixels": int(gain_mask.sum()),
    }
    (OUT_DIR / "facts_2018_2024.json").write_text(json.dumps(facts, indent=2), "utf-8")

    # LLM: pass the exact rendered OG-based images
    side_b64 = img_b64(side_png)
    diff_b64 = img_b64(diff_png)
    llm_facts = {
        "global": {
            "ndvi_mean_2018": facts["ndvi_mean_2018"],
            "ndvi_mean_2024": facts["ndvi_mean_2024"],
            "delta_mean": facts["ndvi_mean_diff"],
            "gain_pixels": facts["gain_pixels"],
            "loss_pixels": facts["loss_pixels"],
            "scale_range": [NDVI_VMIN, NDVI_VMAX]
        },
        "grid": facts["grid"],
        "hotspots": facts["hotspots"],
        "quadrant_diff_means": facts["grid_means_3x3"]["diff"],  # back-compat key
        "thresholds": facts["thresholds"]
    }
    prompt = build_prompt(llm_facts)
    j = chat_vision_json(MODEL, prompt, [side_b64, diff_b64], timeout_s=240)

    # Validate JSON & ban hallucinations
    final_text = None
    if isinstance(j, dict):
        txt_block = " ".join(str(j.get(k,"")) for k in ["spatial","causes"])
        bad = violates_banned(txt_block)
        means = j.get("means", {})
        ok_means = (
            isinstance(means, dict) and
            abs(float(means.get("y2018", 0)) - facts["ndvi_mean_2018"]) < 0.01 and
            abs(float(means.get("y2024", 0)) - facts["ndvi_mean_2024"]) < 0.01 and
            abs(float(means.get("diff", 0))  - facts["ndvi_mean_diff"]) < 0.02
        )
        if (not bad) and ok_means:
            final_text = (
                f"**Trend:** {j.get('trend','mixed')} — mean NDVI "
                f"{float(means.get('y2018')):.3f} → {float(means.get('y2024')):.3f} "
                f"(Δ {float(means.get('diff')):.3f}).\n\n"
                f"**Spatial pattern:** {j.get('spatial','')}\n\n"
                f"**Scale:** {j.get('scale','')}\n\n"
                f"**Interpretation:** {j.get('causes','')}\n\n"
                f"**Next steps:** {j.get('next_steps','')}"
            )

    if final_text is None:
        final_text = deterministic_report(facts)

    out_md = OUT_DIR / "summary_2018_2024.md"
    out_md.write_text(final_text + "\n", encoding="utf-8")

    print("\n=== Mentor Echo Summary (2018 vs 2024) ===\n")
    print(final_text)
    print(f"\nSaved:\n  {out_md}\n  {side_png}\n  {diff_png}\n  {OUT_DIR/'facts_2018_2024.json'}\nModel: {MODEL}")

if __name__ == "__main__":
    main()
