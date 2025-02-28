import cv2
import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  
        self.disappeared = OrderedDict()  
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # Register a new object
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # Deregister an object
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# Load the pre-trained MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Labels for the classes in the COCO dataset, index 15 is 'person'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the video stream
cap = cv2.VideoCapture('test.mp4')

# Counters for people moving up and down
up_count = 0
down_count = 0

# Create an instance of CentroidTracker
ct = CentroidTracker()

# Dictionary to store the previous y-coordinate for each object
previous_y = {}

# Set horizontal line position (center of the screen)
line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    (h, w) = frame.shape[:2]

    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    
    detections = net.forward()

    
    centroids = []

    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        
        if confidence > 0.5:
            
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

          
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)

            
            label = f"Person {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    objects = ct.update(centroids)


    for (object_id, centroid) in objects.items():
        current_y = centroid[1]

       
        if object_id in previous_y:
            prev_y = previous_y[object_id]

       
            if prev_y > line_position and current_y < line_position:
                up_count += 1
                print(f"Person {object_id} moved up.")

           
            elif prev_y < line_position and current_y > line_position:
                down_count += 1
                print(f"Person {object_id} moved down.")

        
        previous_y[object_id] = current_y

   
    cv2.line(frame, (0, line_position), (w, line_position), (0, 0, 255), 2)

    
    cv2.putText(frame, f"Up: {up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Down: {down_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    cv2.imshow("People Counting", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
