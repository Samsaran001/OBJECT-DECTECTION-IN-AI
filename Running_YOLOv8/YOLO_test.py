from ultralytics import YOLO
import cv2
model=YOLO('yolov8n.pt')
results=model('../images/2.jpg',show=True)
cv2.waitkey(0)