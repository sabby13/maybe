import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

# Load model once
model = YOLO("yolov8n.pt")

# Initialize tracker ONCE
tracker = Tracker()

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
    unique_ids = set()

    # 🔥 Timeline storage
    timeline = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    detections.append(
                        ([x1, y1, x2 - x1, y2 - y1], conf, 'person')
                    )

        # Tracking
        tracks = tracker.update(detections, frame)

        people_in_frame = 0
        current_time = frame_count / fps

        for obj in tracks:
            track_id = obj["id"]
            l, t, w, h = obj["bbox"]

            people_in_frame += 1
            unique_ids.add(track_id)

            # 🔥 Timeline update
            if track_id not in timeline:
                timeline[track_id] = {
                    "start": current_time,
                    "end": current_time
                }
            else:
                timeline[track_id]["end"] = current_time

            # Draw bounding box
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)

            # Draw ID
            cv2.putText(frame, f"p{track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f"People: {people_in_frame}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # 🔥 Convert timeline → dataframe
    records = []

    for track_id, times in timeline.items():
        start = times["start"]
        end = times["end"]
        duration = end - start

        records.append({
            "Person": f"p{track_id}",
            "Start": round(start, 2),
            "End": round(end, 2),
            "Duration": round(duration, 2)
        })

    df = pd.DataFrame(records)

    return output_path, df