import numpy as np
import cv2

# Path to dataset
DATA_PATH = "datasets/zhaw_depth_dataset.npy"

# Load dataset
depth_dataset = np.load(DATA_PATH)
print(f"Loaded dataset with shape: {depth_dataset.shape}")

# Playback settings: 600 frames / 20 seconds = 30 fps
fps = 30
delay = int(1000 / fps)

# Loop through frames
for i, frame in enumerate(depth_dataset):
    # Normalize frame for visualization
    frame_vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame_vis = np.uint8(frame_vis)

    # Apply colormap (heatmap style)
    frame_colormap = cv2.applyColorMap(frame_vis, cv2.COLORMAP_JET)

    # Show frame
    cv2.imshow("Depth Playback", frame_colormap)
    print(f"Frame {i+1} / {len(depth_dataset)}")

    # Wait key for playback speed and check quit
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Playback finished.")
