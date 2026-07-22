from ultralytics import YOLO
import cv2
import numpy as np
import gradio as gr

# 調整する設定
MODEL_PATH = "/home/8/uj07518/img_annotation/runs/detect/retrain_02/weights/best.pt"
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

model = YOLO(MODEL_PATH)

def predict_selected(editor_value):
    if editor_value is None or editor_value["composite"] is None:
        return None, "画像を選択してください"
    
    selected_image = editor_value["composite"]

    selected_image_bgr = cv2.cvtColor(
        selected_image,
        cv2.COLOR_RGB2BGR
    )

    results = model.predict(
        source=selected_image_bgr,
        save=False,
        conf=0.15,
        verbose=False
    )

    result = results[0]

    # 最も高い検出結果を取得する例
    if result.boxes is None or len(result.boxes) == 0:
        return selected_image, "液滴を検出できませんでした"
    
    lines = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = result.names[class_id]

        lines.append(f"{label}: {confidence:.2f}")
    
    plotted_bgr = result.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    return plotted_rgb, "\n".join(lines)


with gr.Blocks(title="液滴形状検出") as demo:
    with gr.Row():
        image_editor = gr.ImageEditor(
            label="液滴を選択してください",
            type="numpy",
            sources=["upload"],
            transforms=("crop",),
            brush=False,
            eraser=False,
            layers=False
        )

        output_image = gr.Image(
            label="検出結果",
            type="numpy",
        )

    predict_button = gr.Button("検出する", variant="primary")
    output_text = gr.Textbox(
        label="検出ラベル・確度",
        lines=6,
    )

    predict_button.click(
        fn=predict_selected,
        inputs=image_editor,
        outputs=[output_image, output_text]
    )

demo.launch(share=True)

