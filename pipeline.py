import cv2
import pandas as pd
from ultralytics import YOLO

# Load model once (important)
model = YOLO("yolov8n.pt")  # lightweight model

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Error opening video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output/tracked_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_people_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 YOLO detection
        results = model(frame)

        people_in_frame = 0

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls[0])

                # Class 0 = person
                if cls == 0:
                    people_in_frame += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        total_people_detected += people_in_frame

        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f"People: {people_in_frame}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Simple analytics
    df = pd.DataFrame({
        "Metric": ["Total Frames", "Total People Detected"],
        "Value": [frame_count, total_people_detected]
    })

    return output_path, df