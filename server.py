# File: pi_traffic_server.py

import cv2
import torch
import time
import json
from flask import Flask, jsonify
from threading import Thread

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4

# Global detection result
latest_result = {
    "timestamp": "",
    "vehicles": 0,
    "density": "",
    "signal": "",
    "details": [],
}

def traffic_analysis(labels):
    vehicle_count = sum(1 for label in labels if label in ['car', 'truck', 'bus', 'motorbike'])
    
    if vehicle_count < 5:
        density = "Low"
        signal = "Red"
    elif vehicle_count < 15:
        density = "Medium"
        signal = "Yellow"
    else:
        density = "High"
        signal = "Green"
    
    return vehicle_count, density, signal

def detect_loop():
    global latest_result
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        labels = []
        details = []
        for *xyxy, conf, cls in detections:
            label = model.names[int(cls)]
            if label in ['car', 'truck', 'bus', 'motorbike']:
                labels.append(label)
                details.append({
                    "label": label,
                    "bbox": list(map(int, xyxy)),
                    "confidence": float(conf),
                })

        count, density, signal = traffic_analysis(labels)

        latest_result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vehicles": count,
            "density": density,
            "signal": signal,
            "details": details,
        }

        time.sleep(1)

@app.route("/data")
def get_data():
    return jsonify(latest_result)

if __name__ == '__main__':
    print("Starting detection thread...")
    Thread(target=detect_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
