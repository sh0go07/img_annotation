from ultralytics import YOLO

# load new model
model = YOLO("yolov8s.pt")

# start learning
model.train(data="dataset/data.yaml", epochs=50, imgsz=640, device=0)

print("learning finished")

