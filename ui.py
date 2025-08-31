import cv2
import os
import numpy as np
from ultralytics import YOLO
from detect_obstacle import load_exr_depth

class GuidanceUI:
    def __init__(self, video_path, depth_folder, model_path="yolov8n.pt"):
        self.cap = cv2.VideoCapture(video_path)
        self.depth_folder = depth_folder
        self.model = YOLO(model_path)  # YOLOv8
        self.frame_idx = 0

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        # Depth map filename (0001.exr, 0002.exr…)
        depth_file = os.path.join(self.depth_folder, f"{self.frame_idx+1:04d}.exr")
        depth_map = None
        if os.path.exists(depth_file):
            depth_map = load_exr_depth(depth_file)

        # Run YOLO object detection
        results = self.model(frame, verbose=False)[0]

        closest_obj = None
        closest_distance = float("inf")

        # Draw bounding boxes + depth info
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            label = self.model.names[cls]

            # Compute depth inside bounding box
            distance = None
            if depth_map is not None:
                roi = depth_map[y1:y2, x1:x2]
                if roi.size > 0:
                    distance = np.mean(roi)

                    # Track closest obstacle
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_obj = (x1, y1, x2, y2, label, distance)

            # Draw rectangle + label
            disp_text = label
            if distance is not None:
                disp_text += f" {distance:.1f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, disp_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Generate AI advice
        advice = "Clear Path – proceed forward"
        if closest_obj:
            x1, y1, x2, y2, label, distance = closest_obj
            frame_center = frame.shape[1] // 2
            obj_center = (x1 + x2) // 2

            if obj_center < frame_center:
                direction = "left"
                advice = f"Obstacle: {label} {distance:.1f}m ahead on LEFT → Turn Right"
            else:
                direction = "right"
                advice = f"Obstacle: {label} {distance:.1f}m ahead on RIGHT → Turn Left"

        # Show advice on screen
        cv2.putText(frame, f"AI Advice: {advice}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        self.frame_idx += 1
        return frame, depth_map, advice

    def run(self):
        while True:
            frame, _, _ = self.update_frame()
            if frame is None:
                break

            cv2.imshow("Guidance with AI", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = GuidanceUI("guidance_video.mp4", "depth_maps")
    app.run()
