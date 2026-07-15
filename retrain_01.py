from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")

    model.train(
        data="datasets/data.yaml",
        epochs=50,
        imgsz=640,
        name="retrain_01"
    )

print("retrain finished")

