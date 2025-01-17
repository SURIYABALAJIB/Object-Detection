#For Video
import cv2
import numpy as np
import os
yolo_dir = ''  
weights_path = os.path.join(yolo_dir, "yolov3.weights")
config_path = os.path.join(yolo_dir, "yolov3.cfg")
names_path = os.path.join(yolo_dir, "coco.names")
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers()
if isinstance(output_layer_indices, np.ndarray):
    output_layer_indices = output_layer_indices.flatten()
output_layers = [layer_names[i - 1] for i in output_layer_indices]
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
video_path = 0
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
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
    if cv2.waitKey(1) == 27:  
        break
cap.release()
cv2.destroyAllWindows()
