import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

count = 0

try:
    print("Press Ctrl+C to stop capturing")
    print("Capturing an image every 3 seconds...")
    
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save an image every 3 seconds
        image_name = f"realsense_image_{count}.jpg"
        cv2.imwrite(image_name, color_image)
        print(f"Saved {image_name}")
        count += 1
        
        # Wait for 3 seconds
        time.sleep(3)
            
except KeyboardInterrupt:
    print("\nStopping capture...")
finally:
    # Stop streaming
    pipeline.stop()