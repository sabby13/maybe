import cv2
import pandas as pd
import os

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Error opening video file")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output/tracked_video.mp4"

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔹 Dummy processing (we'll replace later)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Temporary dummy analytics
    data = {
        "Person": ["p1"],
        "Start": [0],
        "End": [frame_count / fps],
        "Duration": [frame_count / fps]
    }

    df = pd.DataFrame(data)

    return output_path, df