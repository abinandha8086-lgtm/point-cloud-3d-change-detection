#!/usr/bin/python3
# ADAPT THESE VALUES FOR YOUR SPECIFIC DEPTH CAMERA
# Common values: Intel RealSense D415/D435, Orbbec Astra, etc.

CAMERA_PARAMS = {
    "width": 640,
    "height": 480,
    "fx": 525.0,    # Focal length X (pixels)
    "fy": 525.0,    # Focal length Y (pixels)  
    "cx": 320.0,    # Principal point X (pixels)
    "cy": 240.0,    # Principal point Y (pixels)
    "depth_scale": 1000.0  # mm per depth unit (RealSense=1000, others vary)
}

# Print for verification
if __name__ == "__main__":
    print("Camera intrinsics loaded:")
    for k, v in CAMERA_PARAMS.items():
        print(f"  {k}: {v}")
