import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO

from ai_worker.ai.roi_extractor import ROIExtractor
from ai_worker.ai.motion_tracker import MotionTracker

from ai_worker.ai.signal_filter import SignalFilter
import time
from ai_worker.ai.breathing_rate import BreathingRateEstimator

model = YOLO("yolo11n-pose.onnx")

roi_extractor = ROIExtractor()

tracker = MotionTracker()

cap = cv2.VideoCapture(0)

fixed_roi = None

frame_count = 0

MAX_FRAMES = 900

print("Bat dau thu du lieu...")
start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if fixed_roi is None:

        result = model(
            frame,
            verbose=False
        )[0]

        if (
            result.keypoints is not None
            and len(result.keypoints.xy) > 0
        ):

            person = result.keypoints.xy[0]

            ls = person[5]
            rs = person[6]
            lh = person[11]
            rh = person[12]

            x1, y1, x2, y2 = roi_extractor.get_chest_roi(
                frame,
                ls,
                rs,
                lh,
                rh
            )

            h, w = frame.shape[:2]

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))

            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 > x1 and y2 > y1:

                fixed_roi = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                print(
                    "ROI da duoc khoa"
                )

    if fixed_roi is not None:

        x1, y1, x2, y2 = fixed_roi

        roi = frame[
            y1:y2,
            x1:x2
        ]

        cv2.imshow(
            "ROI",
            roi
        )

        motion = tracker.process(
            roi
        )

        if motion is not None:

            print(
                f"Motion: {motion:.6f}"
            )

            cv2.putText(
                frame,
                f"Motion: {motion:.6f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Chest ROI",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

    if frame_count >= MAX_FRAMES:
        break


cap.release()
cv2.destroyAllWindows()

duration = time.time() - start_time

print(
    f"Duration: {duration:.2f} seconds"
)
if len(tracker.signal) > 0:

    filtered_signal = SignalFilter.smooth(
        tracker.signal
    )
    bpm, peaks = BreathingRateEstimator.estimate(
        filtered_signal,
        duration
    )

    print(
        f"Respiration Rate: {bpm:.2f} BPM"
    )
    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        tracker.signal,
        label="Raw Signal"
    )

    plt.plot(
        filtered_signal,
        linewidth=2,
        label="Filtered Signal"
    )

    plt.title(
        "Respiration Signal Analysis"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Vertical Motion"
    )

    plt.legend()

    plt.grid(True)

    plt.show()