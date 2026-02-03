import cv2
import numpy as np
# -------------------------------
# Parameters
# -------------------------------
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
# -------------------------------
# Load class names
# -------------------------------
with open("coco.names", "r") as f:
    classNames = f.read().strip().split("\n")
# -------------------------------
# Load model
# -------------------------------
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)
# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    classIds, confs, boxes = net.detect(frame, confThreshold=CONF_THRESHOLD)

    if len(classIds) > 0:
        boxes = boxes.tolist()
        confs = confs.flatten().tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD
        )

        for i in indices.flatten():
            x, y, w, h = boxes[i]

            class_id = int(classIds[i])     # ✅ FIXED
            label = classNames[class_id - 1]
            confidence = int(confs[i] * 100)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
