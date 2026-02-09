from pathlib import Path
import base64
import json
import re

import cv2
import numpy as np
from openai import OpenAI

from groundingdino.util.inference import annotate, load_image, load_model, predict

api_key="YOUR OWN API KEY"
client = OpenAI(api_key=api_key)

dataset_dir = Path("/home/user/heejun/MO506/GroundingDINO/data/mo506_dataset")
results_root = Path("/home/user/heejun/MO506/GroundingDINO/results")
detection_dir = results_root / "detection"
initial_dir = results_root / "initial"
final_dir = results_root / "final"
planning_dir = results_root / "planning"
for directory in (detection_dir, initial_dir, final_dir, planning_dir):
    directory.mkdir(parents=True, exist_ok=True)

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
BOX_TRESHOLD = 0.20  # 0.35
TEXT_TRESHOLD = 0.25

anchor_set = [
    "straight",
    "turn_left",
    "turn_right",
    "change_left_lane",
    "change_right_lane",
    "stop",
]

image_files = sorted(
    [p for p in dataset_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
)

if not image_files:
    raise FileNotFoundError(f"No images found in {dataset_dir}")

for image_path in image_files:
    filename = image_path.stem
    print(f"Processing: {image_path}")

    # 1. chatgpt anomaly detection
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """
The image is from the front camera of an autonomous vehicle.
Analyze the anomaly objects(only objects like animals and materials, not building, snow, rain).
On the first line, list the anomaly objects, separated by periods. If there are duplicate objects, mention each only once.

Here are some examples:

Example 1:
Tiger . Chair . Box .

Example 2:
Kangaroo . Desk .

Example 3:
Nothing
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content

    lines = reply.split("\n")
    anomaly_objects = lines[0] if len(lines) > 0 else ""

    print(anomaly_objects)

    # 2. dino detection
    image_source, image = load_image(str(image_path))
    if anomaly_objects == "Nothing":
        anomaly_objects = ""
    elif "Nothing" in anomaly_objects:
        anomaly_objects = anomaly_objects.replace("Nothing", "")

    dino_prompt = anomaly_objects + " Car . Truck . Traffic light . Pedestraian ."

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=dino_prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    save_path = detection_dir / f"{filename}.png"
    cv2.imwrite(str(save_path), annotated_frame)
    print(save_path)

    # 3. anchor selection
    decision_prompt = f"""
You are an autonomous driving reasoning agent.

You have received the image and the anomaly object detection outputs. 
Your goal is to select the **most appropriate driving action** from the given anchor set.

---

📦 Detected anomaly objects (phrases with confidence):
{[(p, float(l)) for p, l in zip(phrases, logits.tolist())]}

📍 Bounding boxes (normalized xywh):
{boxes.tolist()}

🚦 Available driving anchors:
{anchor_set}

---

Please reason step-by-step based on the object positions (boxes), their type.
and then output **only the final driving action** from the anchor set.
If there is an object on the road but changing the lane is possible, please do so.
Do not explain. Just output one of: {', '.join(anchor_set)}.
"""

    decision_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an autonomous driving decision reasoning model."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": decision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            },
        ],
        temperature=0
    )

    decision = decision_response.choices[0].message.content.strip()
    print("🚘 Predicted driving action:", decision)

    # 4. path planning
    image_for_initial = cv2.imread(str(image_path))
    anchor_traj = {
        'straight' : np.array([
            [960, 1071],
            [960, 983],
            [960, 939],
            [960, 913],
            [960, 899],
            [960, 888]], dtype=np.int32),
        'turn_left' : np.array([
            [1284, 1401],
            [1156, 1309],
            [1087, 1265],
            [987, 1205],
            [960, 1154],
            [1007, 1107]], dtype=np.int32),
        'change_left_lane' : np.array([
            [960, 1060],
            [913, 974],
            [860, 902],
            [816, 862],
            [811, 817],
            [832, 790]], dtype=np.int32),
        'stop' : np.array([
            [960, 1070],
            [960, 1060],
            [960, 1055],
            [960, 1050],
            [960, 1045],
            [960, 1040]], dtype=np.int32),
        
    }

    if decision not in anchor_traj:
        print(f"⚠️ '{decision}' not in predefined anchors, defaulting to 'straight'.")
        decision = "straight"

    points = anchor_traj[decision]
    for i in range(len(points) - 1):
        start = tuple(points[i])
        end = tuple(points[i + 1])
        cv2.line(image_for_initial, start, end, color=(0, 255, 0), thickness=3)  # 초록색 경로

    for idx, (x, y) in enumerate(points):
        color = (0, 0, 255) if idx == 0 else (255, 200, 0)  # 첫 점은 빨강
        cv2.circle(image_for_initial, (x, y), radius=6, color=color, thickness=-1)
        cv2.putText(
            image_for_initial,
            f"t={(idx+1)*0.5:.1f}s",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    save_traj_path = initial_dir / f"{filename}.png"
    cv2.imwrite(str(save_traj_path), image_for_initial)
    print(f"✅ Trajectory visualized and saved to: {save_traj_path}")

    # -------------------------------
    # 🚀 Trajectory Refinement via GPT
    # -------------------------------

    # ✅ 현재 trajectory points를 문자열로 포맷
    points_str = points.tolist()

    # ✅ GPT에게 trajectory refinement 요청
    refine_prompt = f"""
You are an autonomous driving planning agent.

You have received the original trajectory points (image coordinates) from a front camera frame, 
and the selected driving decision (anchor) from a reasoning model.

Your goal: refine or adjust the trajectory points to match the selected decision.
If the decision is 'change to left lane', shift the trajectory smoothly to the left side of the lane.
If 'change to right lane', shift it to the right side.
If 'turn left' or 'turn right', curve it accordingly.
If 'stop', shorten the trajectory near the current position.
If 'forward', adjust only the x values so that the trajectory better follows the center of the currnet road.

---

🧭 Current decision: {decision}

📍 Original trajectory points (from top-left origin):
{points_str}

🖼️ Reference image is provided below. Use it to estimate realistic lane boundaries or curvature if possible.

Please output **only** a JSON list of the refined coordinates in pixel values (same image coordinate system).
Example output format:
[[1234, 1002], [1201, 950], [1180, 900], [1160, 850], [1150, 820], [1130, 800]]
"""

    # ✅ GPT API 호출 (이미지 + 프롬프트 함께)
    refine_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a driving trajectory planner that refines motion paths."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": refine_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            },
        ],
        temperature=0.2
    )

    # ✅ GPT 응답 파싱
    refined_reply = refine_response.choices[0].message.content.strip()
    text = refined_reply.strip()
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    cleaned = cleaned.strip().replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")

    refined_points = np.array(json.loads(cleaned), dtype=np.int32)

    # ✅ refined trajectory 시각화
    refined_image = cv2.imread(str(image_path))

    for i in range(len(refined_points) - 1):
        start = tuple(refined_points[i])
        end = tuple(refined_points[i + 1])
        cv2.line(refined_image, start, end, color=(255, 0, 0), thickness=3)  # 파란색 refined trajectory

    for idx, (x, y) in enumerate(refined_points):
        color = (0, 0, 255) if idx == 0 else (255, 150, 50)
        cv2.circle(refined_image, (x, y), radius=5, color=color, thickness=-1)
        cv2.putText(refined_image, f"r{(idx+1)*0.5:.1f}s", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    save_refined_path = final_dir / f"{filename}.png"
    cv2.imwrite(str(save_refined_path), refined_image)
    print(f"✅ Refined trajectory visualized and saved to: {save_refined_path}")

    planning_text_path = planning_dir / f"{filename}.txt"
    with open(planning_text_path, "w") as plan_file:
        plan_file.write(f"decision: {decision}\n")
        plan_file.write(f"refined_points: {refined_points.tolist()}\n")
    print(f"📝 Planning data saved to: {planning_text_path}")
