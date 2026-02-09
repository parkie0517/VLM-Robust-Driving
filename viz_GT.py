"""
Visualize GT trajectories on top of dataset images.

Reads images from data/mo506_dataset/images, corresponding GT JSON from
data/mo506_dataset/GT, and writes overlays to data/mo506_dataset/images_with_GT.
Only images with existing GT JSON are processed.

Origin: bottom center of the image (x -> right, y -> up).
Each GT entry has r (time), horizontal_distance_px, vertical_distance_px.
We draw a line starting from the origin to the first waypoint, then between
subsequent waypoints; dots at 0.5, 1.0, ..., 3.0s. No dot at the origin.
"""

import json
from pathlib import Path

import cv2

IMAGES_DIR = Path("/home/user/heejun/MO506/GroundingDINO/data/mo506_dataset/images")
GT_DIR = Path("/home/user/heejun/MO506/GroundingDINO/data/mo506_dataset/GT")
OUT_DIR = Path("/home/user/heejun/MO506/GroundingDINO/data/mo506_dataset/images_with_GT")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_points(gt_path: Path):
    """Return sorted list of (r, h_px, v_px) from GT json."""
    with open(gt_path, "r") as f:
        data = json.load(f)

    points = []

    def maybe_add(r_val, entry):
        if r_val is None:
            return
        if not isinstance(entry, dict):
            return
        # Some files nest under "dot"
        dot = entry.get("dot", entry)
        h = dot.get("horizontal_distance_px")
        v = dot.get("vertical_distance_px")
        if h is None or v is None:
            return
        try:
            r_float = float(r_val)
            points.append((r_float, float(h), float(v)))
        except (ValueError, TypeError):
            return

    if isinstance(data, list):
        for ent in data:
            maybe_add(ent.get("r"), ent)
    elif isinstance(data, dict):
        if isinstance(data.get("points"), list):
            for ent in data["points"]:
                maybe_add(ent.get("r"), ent)
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    maybe_add(v.get("r", k), v)

    points.sort(key=lambda x: x[0])
    return points


def draw_gt(image_path: Path, gt_path: Path, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return

    h, w = img.shape[:2]
    origin = (w // 2, h - 1)

    pts = extract_points(gt_path)
    if not pts:
        print(f"⚠️ No valid points in GT: {gt_path}")
        return

    # Convert to image coordinates
    img_pts = []
    for r, h_px, v_px in pts:
        x = int(origin[0] + h_px)
        y = int(origin[1] - v_px)
        img_pts.append((r, (x, y)))

    # Draw polyline from origin to first waypoint, then between waypoints.
    line_color = (0, 255, 0)  # green
    dot_color = (0, 215, 255)  # amber-ish
    thickness = 3

    cv2.line(img, origin, img_pts[0][1], color=line_color, thickness=thickness)
    for i in range(len(img_pts) - 1):
        cv2.line(img, img_pts[i][1], img_pts[i + 1][1], color=line_color, thickness=thickness)

    for r, (x, y) in img_pts:
        cv2.circle(img, (x, y), radius=6, color=dot_color, thickness=-1)
        cv2.putText(
            img,
            f"{r:.1f}s",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.imwrite(str(out_path), img)
    print(f"✅ Saved: {out_path}")


def main():
    images = sorted(
        [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    if not images:
        print(f"❌ No images in {IMAGES_DIR}")
        return

    for img_path in images:
        gt_path = GT_DIR / f"{img_path.stem}_result.json"
        if not gt_path.exists():
            print(f"⏭️ No GT for {img_path.stem}, skipping.")
            continue

        out_path = OUT_DIR / img_path.name
        try:
            draw_gt(img_path, gt_path, out_path)
        except Exception as e:
            print(f"❌ Failed {img_path.stem}: {e}")


if __name__ == "__main__":
    main()
