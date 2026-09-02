from ultralytics import YOLO
import cv2
import numpy as np
import torch

from sort.sort import Sort
from util import get_car, read_license_plate, write_csv


results = {}

# -----------------------------
# GPU / DEVICE
# -----------------------------
device = 0 if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# -----------------------------
# TRACKER
# -----------------------------
mot_tracker = Sort()

# -----------------------------
# LOAD MODELS
# -----------------------------
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# -----------------------------
# LOAD VIDEO
# -----------------------------
video_path = './PORT.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video:", video_path)
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Video opened successfully")
print("Total frames:", total_frames)
print("FPS:", fps)

# COCO classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
vehicles = [2, 3, 5, 7]

# -----------------------------
# READ FRAMES
# -----------------------------
frame_nmr = -1

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_nmr += 1

    # TEST MODE: process only first 5000 frames
    if frame_nmr >= 5000:
        print("\nReached 5000-frame test limit.")
        break

    results[frame_nmr] = {}

    print(f"Processing frame {frame_nmr + 1}/{total_frames}", end='\r')

    # -----------------------------
    # VEHICLE DETECTION
    # -----------------------------
    detections = coco_model(
        frame,
        device=device,
        verbose=False
    )[0]

    detections_ = []

    for detection in detections.boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = detection

        if int(class_id) in vehicles:

            detections_.append([
                x1,
                y1,
                x2,
                y2,
                score
            ])

    # -----------------------------
    # FIX SORT EMPTY-DETECTION ERROR
    # -----------------------------
    if len(detections_) > 0:

        detections_array = np.array(
            detections_,
            dtype=np.float32
        ).reshape(-1, 5)

    else:

        detections_array = np.empty(
            (0, 5),
            dtype=np.float32
        )

    # -----------------------------
    # TRACK VEHICLES
    # -----------------------------
    track_ids = mot_tracker.update(
        detections_array
    )

    # Mask CCTV timestamp area (top-right)
    frame_for_plate = frame.copy()

    h, w = frame_for_plate.shape[:2]

    # Adjust these values if needed
    cv2.rectangle(
        frame_for_plate,
        (int(w * 0.72), 0),
        (w, int(h * 0.15)),
        (0, 0, 0),
        -1
    )

    # -----------------------------
    # LICENSE PLATE DETECTION
    # -----------------------------
    license_plates = license_plate_detector(
        frame,
        device=device,
        conf=0.80,
        imgsz=1280,
        verbose=False
    )[0]

    for license_plate in license_plates.boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = license_plate
        # Ignore CCTV timestamp / top overlay area
    if y2 < 300:
        continue

        # -----------------------------
        # ASSIGN PLATE TO VEHICLE
        # -----------------------------
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(
            license_plate,
            track_ids
        )

        if car_id == -1:
            continue

        # -----------------------------
        # CROP LICENSE PLATE
        # -----------------------------
        license_plate_crop = frame[
            int(y1):int(y2),
            int(x1):int(x2)
        ]

        if license_plate_crop.size == 0:
            continue

        # -----------------------------
        # PREPROCESS PLATE
        # -----------------------------
        license_plate_crop_gray = cv2.cvtColor(
            license_plate_crop,
            cv2.COLOR_BGR2GRAY
        )

        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray,
            64,
            255,
            cv2.THRESH_BINARY_INV
        )

        # -----------------------------
        # OCR
        # -----------------------------
        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_thresh
        )

        if license_plate_text is not None:

            results[frame_nmr][car_id] = {

                'car': {
                    'bbox': [
                        xcar1,
                        ycar1,
                        xcar2,
                        ycar2
                    ]
                },

                'license_plate': {
                    'bbox': [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score': license_plate_text_score
                }
            }

            print(
                f"\nPlate detected | "
                f"Vehicle ID: {int(car_id)} | "
                f"Plate: {license_plate_text} | "
                f"OCR confidence: {license_plate_text_score:.2f}"
            )

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()

print("\nVideo processing completed.")

# -----------------------------
# WRITE RESULTS
# -----------------------------
write_csv(
    results,
    './test_results.csv'
)

print("Results saved to test.csv")