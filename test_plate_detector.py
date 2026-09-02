from ultralytics import YOLO
import cv2
import os
import torch

VIDEO = "./PORT.mp4"
MODEL = "license_plate_detector.pt"

OUTPUT = "./debug_plate_detections"

os.makedirs(OUTPUT, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"

print(
    "Using:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", total_frames)

frame_number = 0
plate_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Test every frame
    result = model(
        frame,
        device=device,
        conf=0.70,
        imgsz=1280,
        verbose=False
    )[0]

    if len(result.boxes) > 0:

        for box in result.boxes.data.tolist():

            x1, y1, x2, y2, confidence, class_id = box

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            plate_count += 1

            filename = (
                f"plate_{plate_count:04d}"
                f"_frame_{frame_number}"
                f"_conf_{confidence:.2f}.jpg"
            )

            cv2.imwrite(
                os.path.join(OUTPUT, filename),
                crop
            )

            print(
                f"\nPlate candidate #{plate_count}"
                f" | Frame {frame_number}"
                f" | Confidence {confidence:.2f}"
            )

    frame_number += 1

    print(
        f"Processing {frame_number}/{total_frames}"
        f" | Plate candidates: {plate_count}",
        end="\r"
    )

cap.release()

print("\n\nFinished")
print("Total plate candidates:", plate_count)
print("Saved in:", OUTPUT)