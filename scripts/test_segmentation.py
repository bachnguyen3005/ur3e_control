import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO('/home/dinh/catkin_ws/src/ur3e_control/scripts/segment.pt', task='detect')

# Perform segmentation
image_path = '/home/dinh/catkin_ws/src/ur3e_control/scripts/realsense_20250416_163650.jpg'
results = model(image_path)

# Load the original image
original_image = cv2.imread(image_path)

# Process segmentation masks
for result in results:
    if result.masks is not None:
        # Get masks tensor (N, H, W) where N is number of masks
        masks = result.masks.data.cpu().numpy()
        
        # Get boxes and class predictions
        boxes = result.boxes.data.cpu().numpy()
        classes = boxes[:, 5].astype(int)
        
        # Create a mask overlay
        overlay = original_image.copy()

        
        # Process each mask
        for mask, box, cls in zip(masks, boxes, classes):
            # Convert mask to uint8
            mask = mask.astype(np.uint8) * 255
            
            # Create a colored mask (you can assign different colors for different classes)
            color = [255,255,255]
            colored_mask = np.zeros_like(original_image)
            colored_mask[mask > 0] = color
            
            # Overlay the mask on the image
            cv2.addWeighted(colored_mask, 0.3, overlay, 0.7, 0, overlay)
            
            # Optionally, draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            # cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), 2)
            
            # Add class label
            label = f'Class: {result.names[int(cls)]}'
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # Save the result
        cv2.imwrite('segmentation_result.jpg', overlay)