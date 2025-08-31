import cv2
import numpy as np
from ultralytics import YOLO
import OpenEXR, Imath, array, numpy as np


# Load YOLO model (small & fast version)
model = YOLO("yolov8n.pt")

def load_exr_depth(path):
    """Load OpenEXR depth map as numpy float32 array"""
    pt = Imath.PixelType(Imath.PixelType.FLOAT)  # 32-bit float
    file = OpenEXR.InputFile(path)

    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read Z (depth) channel
    depth_str = file.channel("R", pt)  # assuming single channel depth in R
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (size[1], size[0])

    return depth

def detect_obstacles(rgb_frame, depth_map, depth_threshold=5.0):
    """
    Run YOLO detection + depth filtering.
    Returns annotated_frame, advice
    """
    results = model.predict(rgb_frame, verbose=False)

    h, w = rgb_frame.shape[:2]
    annotated_frame = rgb_frame.copy()
    centers_x = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Extract depth in the bounding box region
            obj_depth = np.median(depth_map[y1:y2, x1:x2])

            # Only consider if within depth threshold
            if 0 < obj_depth < depth_threshold:
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                centers_x.append((x1 + x2) // 2)

    # Default advice
    advice = "Safe"
    if len(centers_x) > 0:
        avg_x = np.mean(centers_x)
        if avg_x < w // 3:
            advice = "Turn Right"
        elif avg_x > 2 * w // 3:
            advice = "Turn Left"
        else:
            advice = "Turn Either Side"

    return annotated_frame, advice
