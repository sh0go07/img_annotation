from ultralytics import YOLO
import cv2

# load result
model = YOLO("runs/detect/train-6/weights/best.pt")

# judge new images
results = model.predict(source="test.png", save=True)

# show images
res_plotted = results[0].plot()
cv2.imshow("Results", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("検出結果")

