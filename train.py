from ultralytics import YOLO

if __name__ == "__main__":
    # load new model
    model = YOLO("yolov8s.pt")

    # start learning
    model.train(data="datasets/data.yaml", epochs=50, imgsz=640, device=0)

print("learning finished")

