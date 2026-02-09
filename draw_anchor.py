
image_path = "/home/user/heejun/MO506/GroundingDINO/rain_racoon2.png"

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import cv2

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25

# image_source, image = load_image(image_path)

# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption="Racoon . Car . Traffic light . Traffic sign . Truck .",
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )

# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# save_path = "./rain_racoon_dino_result.jpg"
# cv2.imwrite(save_path, annotated_frame)
# print(save_path)



import cv2
import numpy as np


image = cv2.imread(image_path)
points = np.array([
    [515, 575],
    [592, 503],
    [650, 463],
    [682, 426],
    [660, 410],
    [639, 399]
], dtype=np.int32)
 
 
 
 
 
 
# 3️⃣ 각 점을 순서대로 연결 (trajectory)
for i in range(len(points) - 1):
    start = tuple(points[i])
    end = tuple(points[i + 1])
    cv2.line(image, start, end, color=(0, 255, 0), thickness=3)  # 초록색 경로

# 4️⃣ 각 점 표시 (timestamp별 색상 다르게)
for idx, (x, y) in enumerate(points):
    color = (0, 0, 255) if idx == 0 else (255, 200, 0)  # 첫 점은 빨강
    cv2.circle(image, (x, y), radius=6, color=color, thickness=-1)
    cv2.putText(
        image,
        f"t={idx*0.5:.1f}s",
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, # font size
        (255, 255, 255),
        1, # thickness
    )

# 5️⃣ 결과 저장
save_traj_path = "./rain_racoon_trajectory_visualized.png"
cv2.imwrite(save_traj_path, image)
print(f"✅ Trajectory visualized and saved to: {save_traj_path}")

