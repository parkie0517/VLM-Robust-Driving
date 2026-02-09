"""
Compute mean pixel error between GT and predictions.

GT:
 - Path: /home/user/heejun/MO506/GroundingDINO/data/mo506_dataset/GT
 - Files: <stem>_result.json
 - Origin: bottom center of image.
 - Fields: r (timestamp), horizontal_distance_px (x, left negative), vertical_distance_px (y up).

Predictions:
 - Path: /home/user/heejun/MO506/GroundingDINO/results_done/plan
 - Files: <stem>.txt
 - Origin: top-left of image.
 - Second line: refined_points: [[x1,y1], ...] (image coords).
 - If fewer than 6 points, pad by repeating the last point until 6.

Error:
 - Convert GT points to image coords using the target image size (from corresponding image in data/mo506_dataset/images or images_with_GT).
 - Match timestamps [0.5, 1.0, 1.5, 2.0, 2.5, 3.0].
 - Pixel distance per timestamp; sample error = mean of 6 distances.
 - Mean error rate = average of sample errors.
"""

import json
import math
from pathlib import Path

import cv2

GT_DIR = Path("/home/user/heejun/MO506/GroundingDINO/data/mo506_dataset/GT")
PRED_DIR = Path("/home/user/heejun/MO506/GroundingDINO/results_done/plan")
IMAGES_DIR = Path("/home/user/heejun/MO506/GroundingDINO/data/mo506_dataset/images")
FALLBACK_IMG_DIR = Path("/home/user/heejun/MO506/GroundingDINO/data/mo506_dataset/images_with_GT")

TIMESTAMPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_image_shape(stem: str):
    """Return (h, w) for the given stem, looking in images then fallback."""
    for base in (IMAGES_DIR, FALLBACK_IMG_DIR):
        img_path = base / f"{stem}.jpg"
        if not img_path.exists():
            img_path = base / f"{stem}.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                return h, w
    return None


def load_gt(stem: str):
    gt_path = GT_DIR / f"{stem}_result.json"
    if not gt_path.exists():
        return None
    with open(gt_path, "r") as f:
        data = json.load(f)

    pts = {}
    if isinstance(data, list):
        for ent in data:
            r = ent.get("r")
            dot = ent.get("dot", ent)
            h = dot.get("horizontal_distance_px")
            v = dot.get("vertical_distance_px")
            if r is None or h is None or v is None:
                continue
            pts[float(r)] = (float(h), float(v))
    elif isinstance(data, dict):
        items = data.get("points", data)
        if isinstance(items, list):
            iterable = items
        else:
            iterable = items.values()
        for ent in iterable:
            if not isinstance(ent, dict):
                continue
            r = ent.get("r")
            dot = ent.get("dot", ent)
            h = dot.get("horizontal_distance_px")
            v = dot.get("vertical_distance_px")
            if r is None or h is None or v is None:
                continue
            pts[float(r)] = (float(h), float(v))
    return pts


def load_pred(stem: str):
    pred_path = PRED_DIR / f"{stem}.txt"
    if not pred_path.exists():
        return None
    with open(pred_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) < 2:
        return None
    # second line: refined_points: [...]
    rp_line = lines[1]
    prefix = "refined_points:"
    if not rp_line.startswith(prefix):
        return None
    try:
        arr_str = rp_line[len(prefix):].strip()
        pts = json.loads(arr_str)
        if not isinstance(pts, list):
            return None
        pts = [tuple(p) for p in pts]
    except Exception:
        return None
    # pad to 6 by repeating last
    if pts:
        last = pts[-1]
        while len(pts) < 6:
            pts.append(last)
    return pts[:6]


def to_image_coords_from_gt(gt_pts, img_shape):
    """Convert GT coords (h,v) to image coords (x,y) with origin bottom center."""
    h_img, w_img = img_shape
    origin = (w_img / 2.0, h_img - 1.0)
    img_coords = {}
    for r, (h_px, v_px) in gt_pts.items():
        x = origin[0] + h_px
        y = origin[1] - v_px
        img_coords[r] = (x, y)
    return img_coords


def sample_error(stem: str):
    img_shape = load_image_shape(stem)
    if img_shape is None:
        print(f"❌ No image for {stem}")
        return None
    gt_pts = load_gt(stem)
    if not gt_pts:
        print(f"⏭️ No GT for {stem}")
        return None
    pred_pts = load_pred(stem)
    if not pred_pts:
        print(f"⏭️ No prediction for {stem}")
        return None

    gt_img_pts = to_image_coords_from_gt(gt_pts, img_shape)

    # Ensure we have 6 pred points
    if len(pred_pts) < 6 and pred_pts:
        last = pred_pts[-1]
        pred_pts = pred_pts + [last] * (6 - len(pred_pts))
    pred_pts = pred_pts[:6]

    dists = []
    for idx, t in enumerate(TIMESTAMPS):
        gt_xy = gt_img_pts.get(t)
        pred_xy = pred_pts[idx] if idx < len(pred_pts) else None
        if gt_xy is None or pred_xy is None:
            continue
        dx = gt_xy[0] - pred_xy[0]
        dy = gt_xy[1] - pred_xy[1]
        dists.append(math.hypot(dx, dy))

    if len(dists) != len(TIMESTAMPS):
        print(f"⚠️ Missing timestamps for {stem}, computed on {len(dists)} points.")
    return dists if dists else None


def main():
    stems = sorted({p.stem for p in PRED_DIR.glob("*.txt")})
    if not stems:
        print("❌ No prediction files found.")
        return

    errors = []
    per_t_errors = {t: [] for t in TIMESTAMPS}
    for stem in stems:
        dists = sample_error(stem)
        if dists is not None and len(dists) == len(TIMESTAMPS):
            sample_mean = sum(dists) / len(dists)
            errors.append(sample_mean)
            for t, d in zip(TIMESTAMPS, dists):
                per_t_errors[t].append(d)
            print(f"{stem}: error={sample_mean:.2f}")

    if errors:
        mean_err = sum(errors) / len(errors)
        print(f"\nMean error rate over {len(errors)} samples: {mean_err:.2f}")
        for t in TIMESTAMPS:
            if per_t_errors[t]:
                t_mean = sum(per_t_errors[t]) / len(per_t_errors[t])
                print(f"  t={t:.1f}s: mean error={t_mean:.2f}")
    else:
        print("❌ No valid samples to evaluate.")


if __name__ == "__main__":
    main()
