from ultralytics import YOLO
from ultralytics.utils import nms
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 調整する設定
MODEL_PATH = "/home/8/uj07518/img_annotation/runs/detect/train/weights/best.pt"
SOURCE_IMAGE = "test.png"
OUTPUT_IMAGE = "result_multi_label.png"

# YOLO が候補として残す最低確度
PREDICT_CONF = 0.15

# 画像に表示する最低確度
DISPLAY_CONF = 0.15

# 最上位候補との差がこの値以内なら表示する
MAX_CONF_GAP = 0.25

# 最大表示件数
MAX_LABELS = 3

# 同じ物体と判断する枠のIoU
GROUP_IOU = 0.75

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def group_same_object(boxes):
    groups = []

    for row in boxes:
        box = row[:4]
        confidence = float(row[4])
        class_id = int(row[5])

        target_group = None
        for group in groups:
            if calculate_iou(box, group["box"]) >= GROUP_IOU:
                target_group = group
                break
            
        if target_group is None:
            groups.append({
                "box": box,
                "box_confidence": confidence,
                "scores": {class_id: confidence},
            })
        else:
            # 同一クラスが重複した場合は、より高確度な値を残す
            old_score = target_group["scores"].get(class_id, -1)
            target_group["scores"][class_id] = max(old_score, confidence)
    
    return groups

# multi_label = False が標準値のため、実行時だけ True にする
original_nms = nms.non_max_suppression


def multi_label_nms(*args, **kwargs):
    kwargs["multi_label"] = True
    return original_nms(*args, **kwargs)


nms.non_max_suppression = multi_label_nms


# モデル読み込み・推論
model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE_IMAGE,
    save=False,
    conf=PREDICT_CONF,
    agnostic_nms=False,
    verbose=False,
)

result = results[0]
names = result.names

# result.boxes.data: [x1, y1, x2, y2, confidence, class_id]
boxes = result.boxes.data.cpu().numpy()
groups = group_same_object(boxes)

# Pillow での描画設定
image_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(image_pil)


for group in groups:
    x1, y1, x2, y2 = map(int, group["box"])

    # クラス別確度を高い順にする
    ranked = sorted(
        group["scores"].items(),
        key=lambda item: item[1],
        reverse=True
    )

    if not ranked:
        continue

    top_confidence = ranked[0][1]

    # 閾値・最上位との差・最大件数で表示候補を決める
    visible = [
        (class_id, confidence)
        for class_id, confidence in ranked
        if confidence >= DISPLAY_CONF
        and top_confidence - confidence <= MAX_CONF_GAP
    ][:MAX_LABELS]

    text = "\n".join(
        f"{names[class_id]}: {confidence:2f}"
        for class_id, confidence in visible
    )

    # 検出枠
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)

    text_box = draw.multiline_textbbox((x1, y1), text, spacing=4)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    text_x = x1
    text_y = max(0, y1 - text_height - 6)

    draw.rectangle(
        (text_x, text_y, text_x + text_width + 8, text_y + text_height + 6),
        fill=(0, 120, 0)
    )
    draw.multiline_text(
        (text_x + 4, text_y + 2),
        text,
        fill=(255, 255, 255),
        spacing=4
    )


output_rgb = np.array(image_pil)
output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

cv2. imwrite(OUTPUT_IMAGE, output_bgr)
cv2.imshow("results", output_bgr)
cv2.waitKey(0)
cv2.destoryAllWindows()

print(f"検出結果： {OUTPUT_IMAGE}")



