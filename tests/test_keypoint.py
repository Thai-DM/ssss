import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-pose.onnx")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    result = model(frame, verbose=False)[0]

    if result.keypoints is not None:

        points = result.keypoints.xy

        if len(points) > 0:

            person = points[0]

            left_shoulder = person[5]
            right_shoulder = person[6]

            left_hip = person[11]
            right_hip = person[12]

            print(
                "LS:", left_shoulder,
                "RS:", right_shoulder
            )

    cv2.imshow("Pose", result.plot())

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()