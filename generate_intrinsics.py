import json
import numpy as np

def compute_intrinsics(focal_length_mm, sensor_width_mm, sensor_height_mm,
                       image_width_px, image_height_px):
    """
    Compute pinhole camera intrinsics from Blender parameters.
    """
    fx = (focal_length_mm / sensor_width_mm) * image_width_px
    fy = (focal_length_mm / sensor_height_mm) * image_height_px
    cx = image_width_px / 2.0
    cy = image_height_px / 2.0

    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": image_width_px,
        "height": image_height_px,
        "K": [
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ]
    }
    return intrinsics


if __name__ == "__main__":
    # ✏️ Fill these with your Blender camera settings
    focal_length_mm = 24.0       # Camera focal length in mm
    sensor_width_mm = 36.0       # Blender default
    sensor_height_mm = 24.0
    image_width_px = 3840        # Render resolution X
    image_height_px = 2160       # Render resolution Y

    intrinsics = compute_intrinsics(
        focal_length_mm, sensor_width_mm, sensor_height_mm,
        image_width_px, image_height_px
    )

    with open("intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=4)

    print("Saved intrinsics.json with values:")
    print(json.dumps(intrinsics, indent=4))
