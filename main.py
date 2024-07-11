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

# Debugging: Print the output layer indices
print("Output Layer Indices:", output_layer_indices)

# Ensure the indices are in the correct format
if isinstance(output_layer_indices, np.ndarray):
    output_layer_indices = output_layer_indices.flatten()

# Debugging: Print the flattened output layer indices
print("Flattened Output Layer Indices:", output_layer_indices)

output_layers = [layer_names[i - 1] for i in output_layer_indices]

# Debugging: Print the output layers
print("Output Layers:", output_layers)

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
img_path = os.path.join(yolo_dir, "IP1.jpeg")  # Replace with the actual path to your image file
img = cv2.imread(img_path)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
