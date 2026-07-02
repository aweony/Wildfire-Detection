from ultralytics import YOLO
import cv2
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
model = YOLO("yolov8n.pt")

def detect_person_animal(frame, prediction):
    if prediction != "fire" and prediction != "active fire":
        return frame
    
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    class_ids = results[0].boxes.cls.cpu().numpy()  # Get class IDs

    for bbox, class_id in zip(detections, class_ids):
        x1, y1, x2, y2 = map(int, bbox)
        if class_id in [0, 15]:  # Class ID 0 for person and 15 for dog
            label = "Person" if class_id == 0 else "Dog"
            logging.info(f"Detected {label} at [{x1}, {y1}, {x2}, {y2}]")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return frame