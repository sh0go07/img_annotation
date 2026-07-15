from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/retrain_01/weights/best.pt")

    model.train(
        data="datasets/data.yaml",
        epochs=50,
        imgsz=640,
        name="retrain_02"
    )

print("retrain finished")

