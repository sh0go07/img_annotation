import torch
import gradio as gr
import cv2
import numpy as np
import sys
import os
from glob import glob


# import SAM2
sys.path.append("./segment-anything-2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "./segment-anything-2/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# class name
class_list = ["plug", "dripping", "jetting"]

# images path
IMAGE_DIR = "datasets/train/images"
LABEL_DIR = "datasets/train/labels"
os.makedirs(LABEL_DIR, exist_ok=True)

# find images
image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png")))
current_idx = 0

# set the first image
def load_current_image():
    global current_idx, image_paths
    if not image_paths:
        return None, "datasets/train/images の中に写真を入れてね！"

    img_path = image_paths[current_idx]
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # set SAM2
    predictor.set_image(img_rgb)

    status = f"写真 [{current_idx + 1}/{len(image_paths)}]: {os.path.basename(img_path)}"
    return img_rgb, status

# click rule
def get_click_and_mask(img, class_name, evt: gr.SelectData):
    global current_idx, image_paths
    if not image_paths: return img

    class_id = class_list.index(class_name)
    x, y = evt.index

    input_point = np.array([[x, y]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = masks[0]
    y_indices, x_indices = np.where(mask)

    if len(y_indices) > 0 and len(x_indices) > 0:
        h_img, w_img, _ = img.shape
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        w = x_max - x_min
        h = y_max - y_min

        x_center = (x_min + w / 2) / w_img
        y_center = (y_min + h / 2) / h_img
        w_norm = w / w_img
        h_norm = h / h_img

        # make the same name .txt as the images
        img_name = os.path.basename(image_paths[current_idx])
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(LABEL_DIR, txt_name)

        with open(txt_path, "a") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # draw square box
        img_bgr_draw = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        color = (0, 255, 0) if class_id == 0 else ((0, 0, 255) if class_id == 1 else (255, 0, 0))
        cv2.rectangle(img_bgr_draw, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img_bgr_draw, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return cv2.cvtColor(img_bgr_draw, cv2.COLOR_BGR2RGB)
    return img

# push next button
def next_image():
    global current_idx, image_paths
    if current_idx < len(image_paths) - 1:
        current_idx += 1
    return load_current_image()

# application
with gr.Blocks() as demo:
    gr.Markdown("## 👩‍🔬 annotation tool")
    status_label = gr.Label(value="preparing...")
    class_picker = gr.Radio(choices=class_list, value="plug", label="なまえ（クラス）")

    with gr.Row():
        image_input = gr.Image(interactive=False)

    next_btn = gr.Button("next image ➡️")

    # loading set
    demo.load(load_current_image, outputs=[image_input, status_label])
    image_input.select(get_click_and_mask, inputs=[image_input, class_picker], outputs=[image_input])
    next_btn.click(next_image, outputs=[image_input, status_label])

demo.launch(share=True)

