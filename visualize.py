import ast

import cv2
import numpy as np
import pandas as pd


def draw_border(
    img,
    top_left,
    bottom_right,
    color=(0, 255, 0),
    thickness=10,
    line_length_x=200,
    line_length_y=200
):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv('./test_interpolated.csv')


# ============================================================
# LOAD VIDEO
# ============================================================

video_path = 'PORT.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open PORT.mp4")
    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video FPS:", fps)
print("Video Width:", width)
print("Video Height:", height)

out = cv2.VideoWriter(
    './out.mp4',
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PREPARE LICENSE PLATE INFORMATION
# ============================================================

license_plate = {}

for car_id in np.unique(results['car_id']):

    # Get all records for this car
    car_results = results[results['car_id'] == car_id]

    # Find the highest OCR confidence score
    max_score = np.amax(car_results['license_number_score'])

    # Get the row with the best OCR result
    best_result = car_results[
        car_results['license_number_score'] == max_score
    ].iloc[0]

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': best_result['license_number']
    }

    # Move video to the frame containing the best license plate
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        best_result['frame_nmr']
    )

    ret, frame = cap.read()

    if not ret:
        print(f"WARNING: Could not read frame for car {car_id}")
        continue

    # Convert license plate bounding box
    bbox_string = (
        best_result['license_plate_bbox']
        .replace('[ ', '[')
        .replace('   ', ' ')
        .replace('  ', ' ')
        .replace(' ', ',')
    )

    x1, y1, x2, y2 = ast.literal_eval(bbox_string)

    # Crop license plate
    license_crop = frame[
        int(y1):int(y2),
        int(x1):int(x2),
        :
    ]

    # Make sure crop is valid
    if license_crop.size == 0:
        print(f"WARNING: Empty license crop for car {car_id}")
        continue

    # Calculate new width
    crop_height = y2 - y1

    if crop_height <= 0:
        continue

    crop_width = int((x2 - x1) * 400 / crop_height)

    if crop_width <= 0:
        continue

    # Resize license plate
    license_crop = cv2.resize(
        license_crop,
        (crop_width, 400)
    )

    license_plate[car_id]['license_crop'] = license_crop


# ============================================================
# RESET VIDEO TO FIRST FRAME
# ============================================================

frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret = True


# ============================================================
# PROCESS VIDEO FRAME BY FRAME
# ============================================================

print("\nStarting visualization...")
print("Press Q to stop the video.\n")

while ret:

    ret, frame = cap.read()

    frame_nmr += 1

    if not ret:
        break

    # Get results for the current frame
    df_ = results[
        results['frame_nmr'] == frame_nmr
    ]

    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    for row_indx in range(len(df_)):

        row = df_.iloc[row_indx]

        # ----------------------------------------------------
        # CAR BOUNDING BOX
        # ----------------------------------------------------

        car_bbox_string = (
            row['car_bbox']
            .replace('[ ', '[')
            .replace('   ', ' ')
            .replace('  ', ' ')
            .replace(' ', ',')
        )

        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
            car_bbox_string
        )

        draw_border(
            frame,
            (int(car_x1), int(car_y1)),
            (int(car_x2), int(car_y2)),
            (0, 255, 0),
            25,
            line_length_x=200,
            line_length_y=200
        )

        # ----------------------------------------------------
        # LICENSE PLATE BOUNDING BOX
        # ----------------------------------------------------

        plate_bbox_string = (
            row['license_plate_bbox']
            .replace('[ ', '[')
            .replace('   ', ' ')
            .replace('  ', ' ')
            .replace(' ', ',')
        )

        x1, y1, x2, y2 = ast.literal_eval(
            plate_bbox_string
        )

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            12
        )

        # ----------------------------------------------------
        # LICENSE PLATE CROP
        # ----------------------------------------------------

        car_id = row['car_id']

        if car_id not in license_plate:
            continue

        license_crop = license_plate[car_id]['license_crop']

        if license_crop is None:
            continue

        H, W, _ = license_crop.shape

        try:

            # ------------------------------------------------
            # DISPLAY LICENSE PLATE IMAGE
            # ------------------------------------------------

            y_start = int(car_y1) - H - 100
            y_end = int(car_y1) - 100

            x_start = int(
                (car_x2 + car_x1 - W) / 2
            )

            x_end = int(
                (car_x2 + car_x1 + W) / 2
            )

            # Check boundaries
            if (
                y_start >= 0
                and y_end <= frame.shape[0]
                and x_start >= 0
                and x_end <= frame.shape[1]
            ):

                frame[
                    y_start:y_end,
                    x_start:x_end,
                    :
                ] = license_crop

                # --------------------------------------------
                # WHITE BACKGROUND FOR PLATE NUMBER
                # --------------------------------------------

                white_y_start = int(car_y1) - H - 400
                white_y_end = int(car_y1) - H - 100

                if white_y_start >= 0:

                    frame[
                        white_y_start:white_y_end,
                        x_start:x_end,
                        :
                    ] = (255, 255, 255)

                    # ----------------------------------------
                    # LICENSE PLATE NUMBER
                    # ----------------------------------------

                    plate_number = str(
                        license_plate[car_id][
                            'license_plate_number'
                        ]
                    )

                    (
                        text_width,
                        text_height
                    ), _ = cv2.getTextSize(
                        plate_number,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        17
                    )

                    text_x = int(
                        (car_x2 + car_x1 - text_width) / 2
                    )

                    text_y = int(
                        car_y1
                        - H
                        - 250
                        + (text_height / 2)
                    )

                    cv2.putText(
                        frame,
                        plate_number,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        (0, 0, 0),
                        17
                    )

        except Exception:
            pass

    # ========================================================
    # WRITE PROCESSED FRAME TO OUTPUT VIDEO
    # ========================================================

    out.write(frame)

    # ========================================================
    # DISPLAY VIDEO
    # ========================================================

    display_frame = cv2.resize(
        frame,
        (1280, 720)
    )

    cv2.imshow(
        'License Plate Recognition',
        display_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nStopped by user.")
        break


# ============================================================
# CLEAN UP
# ============================================================

out.release()
cap.release()
cv2.destroyAllWindows()

print("\nVisualization completed!")
print("Output video saved as: out.mp4")