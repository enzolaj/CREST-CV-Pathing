"""
This script demonstrates how to open a ZED camera, retrieve color images and depth
visualizations, and display them using OpenCV. It has been updated for compatibility
with ZED SDK 5.0+, focusing on correct API usage, resource management, and clarity.
"""

import pyzed.sl as sl
import cv2
import numpy as np

def main() -> None:
    """
    Initializes and runs the ZED camera stream, displaying color and depth images.
    """
    # 1. Create a Camera object
    zed = sl.Camera()

    # 2. Set initialization parameters
    # This object is used to configure camera settings when opening the device.
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    
    # Corrected: Use 'coordinate_units' attribute, not 'units'
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # 3. Open the camera
    print("Opening ZED camera...")
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        # Use repr() to get a descriptive string representation of the error code
        print(f"Failed to open ZED camera: {repr(status)}")
        zed.close()
        return

    # 4. Prepare data containers and runtime parameters
    # sl.Mat objects are allocated once and reused in the loop for efficiency.
    left_image = sl.Mat()
    depth_image_for_display = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    print("\n--- ZED Camera Stream Running ---")
    print("Press 'q' in the display windows to exit.")

    # 5. Main Capture Loop
    while True:
        # The grab() function captures a new image and runs the depth estimation.
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            
            # --- Retrieve Left Color Image ---
            # retrieve_image() stores the image in the provided sl.Mat object.
            # VIEW.LEFT provides the rectified color image from the left sensor.
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            
            # --- Retrieve Depth Image for Visualization ---
            # retrieve_image() with VIEW.DEPTH provides a human-readable, normalized
            # 8-bit, 4-channel (BGRA) image for display.
            zed.retrieve_image(depth_image_for_display, sl.VIEW.DEPTH)

            # --- Convert to NumPy arrays for OpenCV ---
            # get_data() returns a NumPy array that points to the sl.Mat data.
            # No data is copied, making this a very efficient operation.
            left_image_ocv = left_image.get_data()
            depth_image_ocv = depth_image_for_display.get_data()

            # --- Display Images ---
            cv2.imshow("ZED | Left Color Image", left_image_ocv)
            cv2.imshow("ZED | Depth Visualization", depth_image_ocv)

            # Check for 'q' key press to exit the loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            # If grabbing fails, wait briefly before trying again.
            cv2.waitKey(1)
    
    # 6. Clean up resources
    print("\nClosing the ZED camera and shutting down...")
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()