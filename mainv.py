import cv2
import numpy as np
import os

# Set paths to the YOLO files
yolo_dir = 'D:/pro/object detection'  # Replace with the actual path to your YOLO files
weights_path = os.path.join(yolo_dir, "yolov3.weights")
config_path = os.path.join(yolo_dir, "yolov3.cfg")
names_path = os.path.join(yolo_dir, "coco.names")

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers()

# Ensure the indices are in the correct format
if isinstance(output_layer_indices, np.ndarray):
    output_layer_indices = output_layer_indices.flatten()

output_layers = [layer_names[i - 1] for i in output_layer_indices]

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video
video_path = 0#os.path.join(yolo_dir, "car-detection.mp4")  # Replace with the actual path to your video file
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
