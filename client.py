# File: mac_traffic_dashboard.py

import requests
import cv2
import numpy as np
import time

PI_IP = 'http://<YOUR_PI_IP>:5000/data'  # e.g., http://192.168.1.25:5000/data

def fetch_and_visualize():
    while True:
        try:
            response = requests.get(PI_IP)
            if response.status_code == 200:
                data = response.json()

                # Create blank image
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

                # Draw bounding boxes
                for obj in data['details']:
                    x1, y1, x2, y2 = obj['bbox']
                    label = obj['label']
                    confidence = obj['confidence']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)

                # Overlay analysis
                overlay_texts = [
                    f"Time: {data['timestamp']}",
                    f"Vehicles: {data['vehicles']}",
                    f"Density: {data['density']}",
                    f"Suggested Signal: {data['signal']}",
                ]

                for i, text in enumerate(overlay_texts):
                    cv2.putText(frame, text, (10, 25 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Traffic Dashboard", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print("Error:", e)
        time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    fetch_and_visualize()
