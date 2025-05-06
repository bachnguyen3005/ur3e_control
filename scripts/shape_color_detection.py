import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import math

def detect_shapes_and_colors(image, target_object, roi_offset=(0, 0)):
    # Make a copy of the original image
    original = image.copy()
    result = image.copy()
    all_detected_objects = []
    detected_squares = []
    
    # Store ROI offset
    roi_x, roi_y = roi_offset
    
    # Apply preprocessing to improve detection
    # Multiple blur techniques for better noise reduction
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)  # Preserves edges better
    
    # Convert both to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv_bilateral = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
    
    # Define HSV color ranges with more tolerance
    # Red objects (circle and square) - handling the wrap-around of hue
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([20, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    
    # Blue objects (triangle and square)
    blue_lower = np.array([85, 70, 70])
    blue_upper = np.array([170, 255, 255])
    
    # Improved (broader with lower saturation threshold):
    yellow_lower = np.array([15, 40, 40])  # Lower saturation/value thresholds
    yellow_upper = np.array([50, 255, 255])  # Wider hue range
    
    # Create color masks based on target object
    red_mask = None
    blue_mask = None
    yellow_mask = None
    
    # Create masks for the requested objects
    if target_object in ["red_circle", "red_square", "all"]:
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Try with bilateral filter too
        red_mask1_bi = cv2.inRange(hsv_bilateral, red_lower1, red_upper1)
        red_mask2_bi = cv2.inRange(hsv_bilateral, red_lower2, red_upper2)
        red_mask_bi = cv2.bitwise_or(red_mask1_bi, red_mask2_bi)
        
        # Combine both masks
        red_mask = cv2.bitwise_or(red_mask, red_mask_bi)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    if target_object in ["blue_triangle", "blue_square", "all"]:
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Try with bilateral filter too
        blue_mask_bi = cv2.inRange(hsv_bilateral, blue_lower, blue_upper)
        
        # Combine both masks
        blue_mask = cv2.bitwise_or(blue_mask, blue_mask_bi)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    if target_object in ["yellow_square", "all"]:
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Try with bilateral filter too
        yellow_mask_bi = cv2.inRange(hsv_bilateral, yellow_lower, yellow_upper)
        
        # Combine both masks
        yellow_mask = cv2.bitwise_or(yellow_mask, yellow_mask_bi)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    # Initialize variables for detected objects
    detected_red_circle = None
    detected_blue_triangle = None
    detected_blue_square = None
    detected_red_square = None
    detected_yellow_square = None
    
    # Process specific target object
    if target_object == "red_circle" or target_object == "all":
        if red_mask is not None:
            red_objects = process_color_mask(red_mask, original, "Red", roi_offset)
            for obj in red_objects:
                # Use improved circle detection
                if is_circle(obj):
                    obj['shape'] = 'circle'
                    # Store the original contour for circles
                    obj['original_contour'] = obj['contour']
                    detected_red_circle = obj
                    break
    
    if target_object == "red_square" or target_object == "all":
        if red_mask is not None:
            # Use existing list if we already processed red objects
            if 'red_objects' in locals():
                objects_to_check = red_objects
            else:
                objects_to_check = process_color_mask(red_mask, original, "Red", roi_offset)
                
            for obj in objects_to_check:
                if is_square(obj) and obj['color'] == "Red":
                    obj['shape'] = 'square'
                    detected_red_square = obj
                    break
    
    if target_object == "blue_square" or target_object == "all":
        if blue_mask is not None:
            blue_objects = process_color_mask(blue_mask, original, "Blue", roi_offset)
            for obj in blue_objects:
                if is_blue_square(obj) and obj['color'] == "Blue":
                    obj['shape'] = 'square'
                    detected_blue_square = obj
                    break
    
    if target_object == "yellow_square" or target_object == "all":
        if yellow_mask is not None:
            yellow_objects = process_color_mask(yellow_mask, original, "Yellow", roi_offset)
            for obj in yellow_objects:
                if is_yellow_square(obj) and obj['color'] == "Yellow":
                    obj['shape'] = 'square'
                    detected_yellow_square = obj
                    break
    
    if target_object == "blue_triangle" or target_object == "all":
        if blue_mask is not None:
            # Use existing list if we already processed blue objects
            if 'blue_objects' in locals():
                objects_to_check = blue_objects
            else:
                objects_to_check = process_color_mask(blue_mask, original, "Blue", roi_offset)
                
            # First try standard triangle detection
            for obj in objects_to_check:
                if not objects_are_equal(obj, detected_blue_square) and is_triangle(obj):
                    obj['shape'] = 'triangle'
                    detected_blue_triangle = obj
                    break
            
            # If triangle not found, try specialized detection
            if detected_blue_triangle is None:
                print("Trying specialized triangle detection...")
                blue_triangle_candidates = []
                
                for contour in find_all_contours(blue_mask):
                    area = cv2.contourArea(contour)
                    if area < 50:  # Allow smaller triangles
                        continue
                    
                    # Get shape features
                    shape_info = get_shape_features(contour)
                    
                    # Calculate center of contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    
                    # Adjust center coordinates for ROI offset
                    global_cx = cx + roi_x
                    global_cy = cy + roi_y
                    
                    # Check triangle-specific features
                    has_triangle_shape = (shape_info['vertices'] == 3 or 
                                         (shape_info['vertices'] <= 5 and 
                                          shape_info['compactness'] > 1.2 and
                                          shape_info['compactness'] < 1.8) or
                                         is_triangular(contour))
                    
                    if has_triangle_shape:
                        obj = {
                            'color': "Blue",
                            'shape': 'triangle',
                            'area': area,
                            'approx': shape_info['approx'],
                            'contour': contour,
                            'cx': cx,  # Local coordinate
                            'cy': cy,  # Local coordinate
                            'global_cx': global_cx,  # Global coordinate
                            'global_cy': global_cy,  # Global coordinate
                            'vertices': shape_info['vertices'],
                            'circularity': shape_info['circularity'],
                            'compactness': shape_info['compactness'],
                            'solidity': shape_info.get('solidity', 0),
                            'aspect_ratio': shape_info['aspect_ratio'],
                            'shape_confidence': shape_info.get('shape_confidence', 0),
                            'roi_x': roi_x,  # Store ROI offset
                            'roi_y': roi_y
                        }
                        blue_triangle_candidates.append(obj)
                
                # Select the best triangle candidate if any
                if blue_triangle_candidates:
                    # Sort by decreasing compactness (triangles have distinct compactness)
                    blue_triangle_candidates.sort(key=lambda x: x['compactness'], reverse=True)
                    
                    # Select candidate with best triangle properties
                    best_candidate = None
                    best_score = 0
                    
                    for candidate in blue_triangle_candidates:
                        score = 0
                        # Triangles typically have 3 vertices
                        if candidate['vertices'] == 3:
                            score += 3
                        # Triangles have distinctive compactness
                        if 1.2 < candidate['compactness'] < 1.8:
                            score += 2
                        # Triangles have lower circularity
                        if candidate['circularity'] < 0.8:
                            score += 1
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = candidate
                    
                    if best_candidate:
                        detected_blue_triangle = best_candidate
    
    # Get full image dimensions
    h, w = image.shape[:2]
    
    # Add a reference indicator for global coordinates
    if roi_x > 0 or roi_y > 0:
        reference_text = f"ROI origin: ({roi_x}, {roi_y}) in {w + roi_x}x{h + roi_y} image"
        cv2.putText(result, reference_text, (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Now draw the detected object based on target selection
    if target_object == "red_circle" and detected_red_circle:
        draw_contour(result, detected_red_circle)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Red circle: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_red_circle)

    elif target_object == "blue_triangle" and detected_blue_triangle:
        draw_contour(result, detected_blue_triangle)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Blue triangle: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_blue_triangle)
    
    elif target_object == "blue_square" and detected_blue_square:
        draw_contour(result, detected_blue_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Blue square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_blue_square)
    
    elif target_object == "red_square" and detected_red_square:
        draw_contour(result, detected_red_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Red square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Display red square center
        display_center_coordinates(result, detected_red_square)
    
    # Updated section - draw yellow square if detected
    elif target_object == "yellow_square" and detected_yellow_square:
        draw_contour(result, detected_yellow_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Yellow square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Display yellow square center
        display_center_coordinates(result, detected_yellow_square)
        yaw_angle = calculate_center_grasp_orientation(detected_yellow_square)
        
        # Add visualization to the result image
        result = visualize_center_grasp(result, detected_yellow_square, yaw_angle)
        print(f"Yellow square center grasp yaw angle: {yaw_angle} degrees")
    
    elif target_object == "all":
        # Draw all detected objects
        if detected_blue_square:
            draw_contour(result, detected_blue_square)
            display_center_coordinates(result, detected_blue_square)
            all_detected_objects.append(detected_blue_square)
            
        if detected_blue_triangle:
            draw_contour(result, detected_blue_triangle)
            display_center_coordinates(result, detected_blue_triangle)
            all_detected_objects.append(detected_blue_triangle)

        if detected_red_circle:
            draw_contour(result, detected_red_circle)
            display_center_coordinates(result, detected_red_circle)
            all_detected_objects.append(detected_red_circle)

        if detected_red_square:
            draw_contour(result, detected_red_square)
            # Display red square center
            display_center_coordinates(result, detected_red_square)
            all_detected_objects.append(detected_red_square)
            
        if detected_yellow_square:
            draw_contour(result, detected_yellow_square)
            # Display yellow square center
            display_center_coordinates(result, detected_yellow_square)
            all_detected_objects.append(detected_yellow_square)
    
    # If not found, show status as Not Found
    if target_object == "red_circle" and not detected_red_circle:
        cv2.putText(result, "Red circle: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "blue_triangle" and not detected_blue_triangle:
        cv2.putText(result, "Blue triangle: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "blue_square" and not detected_blue_square:
        cv2.putText(result, "Blue square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "red_square" and not detected_red_square:
        cv2.putText(result, "Red square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "yellow_square" and not detected_yellow_square:
        cv2.putText(result, "Yellow square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Filter squares from all_detected_objects
    for obj in all_detected_objects:
        if obj['shape'] == 'square':
            # Extract color and coordinates (using global coordinates that account for ROI)
            color = obj['color'].lower()  # Convert to lowercase for consistency
            cx = obj['global_cx']
            cy = obj['global_cy']
            
            # Add to detected_squares list as a tuple (x, y, color)
            detected_squares.append((cx, cy, color))
            
    
    return result, detected_squares

def detect_shapes_and_colors_yolo(image, target_object, model_path="/home/dinh/catkin_ws/src/ur3e_control/scripts/object_detection.pt", conf_threshold=0.75):
    # Make a copy of the original image
    original = image.copy()
    result = image.copy()
    all_detected_objects = []
    detected_squares = []
    detected_circle = []
    # Load the YOLO model
    model = YOLO(model_path, task='detect')
    
    # Run inference on the image
    results = model(image, conf=conf_threshold)[0]
    print("DEBUG result: ", results)
    
    # Process results to extract detected objects
    detected_red_circle = None
    detected_blue_triangle = None
    detected_blue_square = None
    detected_red_square = None
    detected_yellow_square = None
    
    # Extract detections from YOLO results
    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class name from the YOLO model
        class_name = results.names[int(cls)]
        
        # Calculate center and area
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Extract color and shape from class_name (assuming format like "red_circle")
        # Check if space is in class name (e.g., "Blue square")
        if " " in class_name:
            color, shape = class_name.split(" ", 1)
            # Preserve original capitalization from YOLO model
        else:
            # Fallback for unexpected format
            color = "Unknown"
            shape = class_name
        
        # Create contour from bounding box for consistency with original function
        contour = np.array([
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]]
        ], dtype=np.int32)
        
        # Create object dict with same structure as original function
        obj = {
            'color': color,
            'shape': shape,
            'area': area,
            'contour': contour,
            'approx': contour,  # Use same contour as approximation for simplicity
            'cx': cx,
            'cy': cy,
            'global_cx': cx,  # No ROI in this function, so local = global
            'global_cy': cy,
            'vertices': 4,  # Default for bounding box
            'circularity': 0.0,  # Not calculated for YOLO
            'compactness': 0.0,  # Not calculated for YOLO
            'solidity': 1.0,  # Not calculated for YOLO
            'aspect_ratio': width / height if height != 0 else 1.0,
            'confidence': conf  # YOLO confidence score
        }
        
        # Add to detected objects based on class
        if class_name == "Red circle" and (target_object == "red_circle" or target_object == "all"):
            detected_red_circle = obj
            all_detected_objects.append(obj)
        
        elif class_name == "Blue triangle" and (target_object == "blue_triangle" or target_object == "all"):
            detected_blue_triangle = obj
            all_detected_objects.append(obj)
        
        elif class_name == "Blue square" and (target_object == "blue_square" or target_object == "all"):
            detected_blue_square = obj
            all_detected_objects.append(obj)
        
        elif class_name == "Red square" and (target_object == "red_square" or target_object == "all"):
            detected_red_square = obj
            all_detected_objects.append(obj)
        
        elif class_name == "Yellow square" and (target_object == "yellow_square" or target_object == "all"):
            detected_yellow_square = obj
            all_detected_objects.append(obj)
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Draw detections based on target_object
    if target_object == "red_circle" and detected_red_circle:
        draw_contour(result, detected_red_circle)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Red circle: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_red_circle)

    elif target_object == "blue_triangle" and detected_blue_triangle:
        draw_contour(result, detected_blue_triangle)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Blue triangle: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_blue_triangle)
    
    elif target_object == "blue_square" and detected_blue_square:
        draw_contour(result, detected_blue_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Blue square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_blue_square)
    
    elif target_object == "red_square" and detected_red_square:
        draw_contour(result, detected_red_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Red square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_red_square)
    
    elif target_object == "yellow_square" and detected_yellow_square:
        draw_contour(result, detected_yellow_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Yellow square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_yellow_square)
        
    
    elif target_object == "all":
        # Draw all detected objects
        if detected_blue_square:
            draw_contour(result, detected_blue_square)
            display_center_coordinates(result, detected_blue_square)
            
        if detected_blue_triangle:
            draw_contour(result, detected_blue_triangle)
            display_center_coordinates(result, detected_blue_triangle)

        if detected_red_circle:
            draw_contour(result, detected_red_circle)
            display_center_coordinates(result, detected_red_circle)

        if detected_red_square:
            draw_contour(result, detected_red_square)
            display_center_coordinates(result, detected_red_square)
            
        if detected_yellow_square:
            draw_contour(result, detected_yellow_square)
            display_center_coordinates(result, detected_yellow_square)
    
    # Handle object not found cases
    if target_object == "red_circle" and not detected_red_circle:
        cv2.putText(result, "Red circle: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "blue_triangle" and not detected_blue_triangle:
        cv2.putText(result, "Blue triangle: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "blue_square" and not detected_blue_square:
        cv2.putText(result, "Blue square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "red_square" and not detected_red_square:
        cv2.putText(result, "Red square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "yellow_square" and not detected_yellow_square:
        cv2.putText(result, "Yellow square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    

    print("All detected objects:", [(obj['color'], obj['shape']) for obj in all_detected_objects])
    
    # Extract detected squares for return value
    for obj in all_detected_objects:
        if 'square' in obj['shape'].lower():
            # Extract color and coordinates
            color = obj['color'].lower()  # Convert to lowercase for consistency
            cx = obj['global_cx']
            cy = obj['global_cy']
            
            # Add to detected_squares list as a tuple (x, y, color)
            detected_squares.append((cx, cy, color))
            print(f"Added square: {color} at ({cx}, {cy})")
            
        if 'circle' in obj['shape'].lower():
            # Extract color and coordinates
            color = obj['color'].lower()  # Convert to lowercase for consistency
            cx = obj['global_cx']
            cy = obj['global_cy']
            
            # Add to detected_squares list as a tuple (x, y, color)
            detected_circle.append((cx, cy, color))
            print(f"Added circle: {color} at ({cx}, {cy})")
    
    print("Extracted squares:", detected_squares)
    
    return result, detected_squares, detected_circle

def detect_shapes_and_colors_yolo_seg(image, target_object, model_path="/home/dinh/catkin_ws/src/ur3e_control/scripts/segment.pt", conf_threshold=0.75):
    # Make a copy of the original image
    original = image.copy()
    result = image.copy()
    all_detected_objects = []
    detected_squares = []
    detected_circle = []
    
    # Load the YOLO segmentation model
    model = YOLO(model_path, task='segment')
    
    # Run inference on the image
    results = model(image, conf=conf_threshold)[0]
    print("DEBUG result: ", results)
    
    # Process results to extract detected objects with segmentation
    detected_red_circle = None
    detected_blue_triangle = None
    detected_blue_square = None
    detected_red_square = None
    detected_yellow_square = None
    
    def calculate_square_rotation(contour):
        """
        Calculate the rotation angle of a square based on its contour.
        Returns angle in degrees between -90 and 90.
        """
        # Find the minimum area rectangle that encloses the contour
        rect = cv2.minAreaRect(contour)
        
        # Get the angle from the rectangle
        # OpenCV's minAreaRect returns angles in the range [-90, 0)
        angle = rect[2]
        
        # Get the width and height of the rectangle
        width, height = rect[1]
        
        # Adjust the angle based on the rectangle's orientation
        # We want to determine if the rectangle is rotated more horizontally or vertically
        if width < height:
            # If width is less than height, adjust the angle
            angle = angle - 90
        
        # Normalize angle to be between -45 and 45 degrees for square rotation
        # (since a square rotated by 90 degrees looks the same)
        if angle < -45:
            angle = angle + 90
        elif angle > 45:
            angle = angle - 90
            
        return angle, rect
    
    # Extract detections from YOLO results
    if results.masks is not None:
        for i, (mask, box) in enumerate(zip(results.masks.data, results.boxes.data)):
            # Extract box coordinates and class
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class name from the YOLO model
            class_name = results.names[int(cls)]
            
            # Calculate center and area based on box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Extract color and shape from class_name
            if " " in class_name:
                color, shape = class_name.split(" ", 1)
            else:
                color = "Unknown"
                shape = class_name
            
            # Convert mask to numpy array for contour extraction
            mask_np = mask.cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            
            # Find contours from the mask for more accurate shape representation
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use the largest contour
                contour = max(contours, key=cv2.contourArea)
                # Calculate rotation angle for shapes
                rotation_angle = 0
                min_area_rect = None
                if 'square' in shape.lower():
                    rotation_angle, min_area_rect = calculate_square_rotation(contour)
                    
                # Calculate properties based on the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    # Use contour centroid for more precise center
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                
                # Calculate circularity for better shape identification
                perimeter = cv2.arcLength(contour, True)
                circularity = 0.0
                if perimeter > 0:
                    circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
                
                # Approximate contour for vertex count
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                # Calculate additional shape properties
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
                
                # Create object dict with enhanced properties from segmentation
                obj = {
                    'color': color,
                    'shape': shape,
                    'area': area,
                    'contour': contour,
                    'approx': approx,
                    'cx': cx,
                    'cy': cy,
                    'global_cx': cx,
                    'global_cy': cy,
                    'vertices': vertices,
                    'circularity': circularity,
                    'compactness': 0.0,  # Not calculated but kept for compatibility
                    'solidity': solidity,
                    'aspect_ratio': width / height if height != 0 else 1.0,
                    'confidence': conf,
                    'mask': mask_uint8,  # Store the segmentation mask
                    'rotation_angle': rotation_angle,
                    'min_area_rect': min_area_rect
                }
                
                # Add to detected objects based on class
                if class_name == "Red circle" and (target_object == "red_circle" or target_object == "all"):
                    detected_red_circle = obj
                    all_detected_objects.append(obj)
                
                elif class_name == "Blue triangle" and (target_object == "blue_triangle" or target_object == "all"):
                    detected_blue_triangle = obj
                    all_detected_objects.append(obj)
                
                elif class_name == "Blue square" and (target_object == "blue_square" or target_object == "all"):
                    detected_blue_square = obj
                    all_detected_objects.append(obj)
                
                elif class_name == "Red square" and (target_object == "red_square" or target_object == "all"):
                    detected_red_square = obj
                    all_detected_objects.append(obj)
                
                elif class_name == "Yellow square" and (target_object == "yellow_square" or target_object == "all"):
                    detected_yellow_square = obj
                    all_detected_objects.append(obj)
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Modified draw_contour function for segmentation results
    def draw_segmentation(img, obj):
        # Draw the segmentation mask
        mask_overlay = img.copy()
        if 'mask' in obj:
            # Create color for the mask based on the object color with higher opacity
            if obj['color'].lower() == 'red':
                color_bgr = (0, 0, 255)  # BGR for red
            elif obj['color'].lower() == 'blue':
                # Make blue more visible against dark background
                color_bgr = (255, 128, 0)  # Brighter blue-orange
            elif obj['color'].lower() == 'yellow':
                color_bgr = (0, 255, 255)  # BGR for yellow
            else:
                color_bgr = (0, 255, 0)  # Default green
            
            # Create colored mask with higher opacity
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask_resized = cv2.resize(obj['mask'], (w, h))
            colored_mask[mask_resized > 0] = color_bgr
            
            # Increase opacity for better visibility (0.7 instead of 0.5)
            cv2.addWeighted(colored_mask, 0.3, mask_overlay, 0.7, 0, mask_overlay)
            
            # Draw thicker contour outline
            cv2.drawContours(mask_overlay, [obj['contour']], -1, color_bgr, 3)
            
            # Add a white border around the mask for better contrast
            kernel = np.ones((5,5), np.uint8)
            dilated_mask = cv2.dilate(mask_resized, kernel, iterations=1)
            border_mask = dilated_mask - mask_resized
            mask_overlay[border_mask > 0] = (255, 255, 255)  # White border
            
            # Draw center point with larger size
            cv2.circle(mask_overlay, (obj['cx'], obj['cy']), 7, (255, 255, 255), -1)
            
            if 'square' in obj['shape'].lower() and obj['min_area_rect'] is not None:
                # Get the rotated rectangle
                box = cv2.boxPoints(obj['min_area_rect']).astype(np.int32)
                
                # Draw the rotated rectangle
                cv2.drawContours(mask_overlay, [box], 0, (0, 255, 255), 2)
                
                # Draw a line indicating the orientation
                center = (obj['cx'], obj['cy'])
                angle_rad = np.radians(obj['rotation_angle'])
                length = 50  # Length of the orientation line
                end_point = (
                    int(center[0] + length * np.cos(angle_rad)),
                    int(center[1] + length * np.sin(angle_rad))
                )
                cv2.line(mask_overlay, center, end_point, (255, 255, 255), 2)
                
                # Add rotation text
                text = f"{obj['rotation_angle']:.1f}°"
                cv2.putText(mask_overlay, text, 
                        (center[0] - 20, center[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                return mask_overlay
            
            return mask_overlay
    
        
        else:
            # Fallback to original contour drawing if no mask
            cv2.drawContours(img, [obj['contour']], -1, (0, 255, 0), 2)
            cv2.circle(img, (obj['cx'], obj['cy']), 5, (255, 255, 255), -1)
            return img
        

    
    # Function to display center coordinates similar to original
    def display_center_coordinates(img, obj):
        text = f"{obj['color']} {obj['shape']}: ({obj['cx']}, {obj['cy']})"
        cv2.putText(img, text, (obj['cx'] + 10, obj['cy']), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw detections based on target_object
    if target_object == "red_circle" and detected_red_circle:
        result = draw_segmentation(result, detected_red_circle)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Red circle: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_red_circle)

    elif target_object == "blue_triangle" and detected_blue_triangle:
        result = draw_segmentation(result, detected_blue_triangle)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Blue triangle: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_blue_triangle)
    
    elif target_object == "blue_square" and detected_blue_square:
        result = draw_segmentation(result, detected_blue_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Blue square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_blue_square)
    
    elif target_object == "red_square" and detected_red_square:
        result = draw_segmentation(result, detected_red_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Red square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_red_square)
    
    elif target_object == "yellow_square" and detected_yellow_square:
        result = draw_segmentation(result, detected_yellow_square)
        status = "Found"
        color = (0, 255, 0)  # Green for found
        cv2.putText(result, f"Yellow square: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        display_center_coordinates(result, detected_yellow_square)
    
    elif target_object == "all":
        # Draw all detected objects
        if detected_blue_square:
            result = draw_segmentation(result, detected_blue_square)
            display_center_coordinates(result, detected_blue_square)
            
        if detected_blue_triangle:
            result = draw_segmentation(result, detected_blue_triangle)
            display_center_coordinates(result, detected_blue_triangle)

        if detected_red_circle:
            result = draw_segmentation(result, detected_red_circle)
            display_center_coordinates(result, detected_red_circle)

        if detected_red_square:
            result = draw_segmentation(result, detected_red_square)
            display_center_coordinates(result, detected_red_square)
            
        if detected_yellow_square:
            result = draw_segmentation(result, detected_yellow_square)
            display_center_coordinates(result, detected_yellow_square)
    
    # Handle object not found cases (same as original)
    if target_object == "red_circle" and not detected_red_circle:
        cv2.putText(result, "Red circle: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "blue_triangle" and not detected_blue_triangle:
        cv2.putText(result, "Blue triangle: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "blue_square" and not detected_blue_square:
        cv2.putText(result, "Blue square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "red_square" and not detected_red_square:
        cv2.putText(result, "Red square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif target_object == "yellow_square" and not detected_yellow_square:
        cv2.putText(result, "Yellow square: Not Found", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    print("All detected objects:", [(obj['color'], obj['shape']) for obj in all_detected_objects])
    
    # Extract detected squares and circles for return value (same as original)
    for obj in all_detected_objects:
        if 'square' in obj['shape'].lower():
            color = obj['color'].lower()
            cx = obj['global_cx']
            cy = obj['global_cy']
            rotation = obj['rotation_angle']
            detected_squares.append((cx, cy, color, rotation))
            print(f"Added square: {color} at ({cx}, {cy}), rotation: {rotation:.1f}")
            
        if 'circle' in obj['shape'].lower():
            color = obj['color'].lower()
            cx = obj['global_cx']
            cy = obj['global_cy']
            detected_circle.append((cx, cy, color))
            print(f"Added circle: {color} at ({cx}, {cy})")
    
    print("Extracted squares:", detected_squares)
    
    return result, detected_squares, detected_circle

def display_center_coordinates(image, obj):
    """Display the center coordinates of an object on the image with improved visibility.
    Shows both local (ROI) coordinates and global (whole image) coordinates."""
    if obj is None:
        return
    
    # Get center coordinates (local to the ROI)
    cx, cy = obj['cx'], obj['cy']
    
    # Get global coordinates (relative to the whole image)
    global_cx = obj.get('global_cx', cx)  # Default to local if not available
    global_cy = obj.get('global_cy', cy)  # Default to local if not available
    
    # Get ROI offset
    roi_x = obj.get('roi_x', 0)  # Default to 0 if not available
    roi_y = obj.get('roi_y', 0)  # Default to 0 if not available
    
    # Draw a more visible marker at the center
    cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
    
    # Create background for better text visibility - display global coordinates
    text = f"Center: ({global_cx}, {global_cy})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    print(f"{obj['color']} {obj['shape']} is at: {text}")
    
    # Shift the text placement: down by 15 pixels and left by 20 pixels
    text_x = cx - 100  # Shifted left (was cx + 10)
    text_y = cy + 15  # Shifted down
    
    # Create background rectangle for text, adjusted for new position
    cv2.rectangle(image,
                 (text_x, text_y - 5),
                 (text_x + text_size[0], text_y + text_size[1] + 5),
                 (0, 0, 0),
                 -1)
                 
    # Display the coordinates text, adjusted for new position
    cv2.putText(image, text, (text_x, text_y + text_size[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def is_square(obj):
    """Check if an object is a square based on multiple features"""
    return ((obj['vertices'] >= 4 and obj['vertices'] <= 6) and 
            0.7 <= obj['aspect_ratio'] <= 1.3 and 
            obj['circularity'] < 0.87 and
            obj['compactness'] < 1.3)

def objects_are_equal(obj1, obj2):
    """Compare two objects to check if they are the same"""
    if obj1 is None or obj2 is None:
        return False
    
    # Compare centers and areas as a simple check
    area_match = abs(obj1['area'] - obj2['area']) < (obj1['area'] * 0.1)  # 10% tolerance
    center_match = (abs(obj1['cx'] - obj2['cx']) < 10 and  # 10 pixel tolerance
                   abs(obj1['cy'] - obj2['cy']) < 10)
    
    return area_match and center_match

def find_all_contours(mask):
    """Find all contours in a mask, regardless of hierarchy"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_shape_features(contour):
    """Extract more robust shape features from a contour"""
    # Calculate perimeter and area
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    # Different epsilon values for different approximation needs
    approximations = []
    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        approximations.append((len(approx), approx, epsilon_factor))
    
    # Sort approximations by number of vertices (ascending)
    approximations.sort(key=lambda x: x[0])
    
    # Get the most suitable approximation
    # For shapes with expected vertex counts, pick the closest match
    vertices_3_to_6 = [a for a in approximations if 3 <= a[0] <= 6]
    if vertices_3_to_6:
        best_approx = vertices_3_to_6[0][1]  # Take first (simplest) approximation with 3-6 vertices
        best_vertices = vertices_3_to_6[0][0]
    else:
        # Default to the simplest reasonable approximation
        best_approx = approximations[0][1] if approximations[0][0] >= 3 else approximations[-1][1]
        best_vertices = len(best_approx)
    
    # Calculate improved circularity (4π*area/perimeter²)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    
    # Get convex hull and its properties
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    # Calculate solidity (area/hull_area) - how much the shape fills its convex hull
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Calculate bounding box properties
    x, y, w, h = cv2.boundingRect(best_approx)
    aspect_ratio = float(w) / h if h > 0 else 0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0  # How much the contour fills its bounding rectangle
    
    # Calculate rotated rectangle properties for more precise aspect ratio
    rot_rect = cv2.minAreaRect(contour)
    rot_w, rot_h = rot_rect[1]
    rot_aspect_ratio = max(rot_w, rot_h) / min(rot_w, rot_h) if min(rot_w, rot_h) > 0 else 0
    
    # Calculate minimum enclosing circle metrics
    (x, y), radius = cv2.minEnclosingCircle(contour)
    min_circle_area = np.pi * radius * radius
    circle_similarity = area / min_circle_area if min_circle_area > 0 else 0
    
    # Calculate hull perimeter and compactness (hull perimeter^2 / (4π * hull area))
    hull_peri = cv2.arcLength(hull, True)
    compactness = hull_peri ** 2 / (4 * np.pi * hull_area) if hull_area > 0 else 0
    
    # Determine shape based on improved geometric properties
    shape = "unknown"
    shape_confidence = 0  # 0-1 scale for confidence in shape determination
    
    # Improved triangle detection
    triangle_score = 0
    if best_vertices == 3:
        triangle_score += 0.7
    if 1.2 < compactness < 1.8:  # Triangles have distinct compactness
        triangle_score += 0.2
    if solidity > 0.85:  # Triangles tend to fill their hull well
        triangle_score += 0.1
    
    # Improved square/rectangle detection
    square_score = 0
    if best_vertices == 4:
        square_score += 0.5
    if 0.8 < aspect_ratio < 1.2:  # Square: aspect ratio close to 1
        square_score += 0.3
    elif 0.2 < aspect_ratio < 0.8 or 1.2 < aspect_ratio < 5:  # Rectangle: clear elongation
        square_score += 0.1
    if extent > 0.8:  # Squares/rectangles fill their bounding box well
        square_score += 0.1
    if solidity > 0.95:  # Squares/rectangles are highly convex
        square_score += 0.1
    
    # Improved circle detection
    circle_score = 0
    if circularity > 0.85:  # High circularity is primary circle indicator
        circle_score += 0.7
    if 0.9 < rot_aspect_ratio < 1.1:  # Circles have balanced aspect ratio
        circle_score += 0.1
    if circle_similarity > 0.85:  # Circles fill their min enclosing circle well
        circle_score += 0.2
    
    # Assign shape based on highest score
    scores = [("triangle", triangle_score), ("square", square_score), ("circle", circle_score)]
    best_shape, best_score = max(scores, key=lambda x: x[1])
    
    if best_score > 0.6:  # Only assign a shape if we have reasonable confidence
        shape = best_shape
        shape_confidence = best_score
    else:
        shape = f"{best_vertices}-sided"
    
    return {
        'shape': shape,
        'approx': best_approx,
        'vertices': best_vertices,
        'circularity': circularity,
        'compactness': compactness,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'rot_aspect_ratio': rot_aspect_ratio,
        'circle_similarity': circle_similarity,
        'shape_confidence': shape_confidence
    }

def is_triangle(obj):
    """Improved triangle detection using multiple metrics"""
    # Strong triangle indicators
    if obj['vertices'] == 3 and obj['solidity'] > 0.85:
        return True
    
    # Reasonable triangle indicators with partial evidence
    if ((obj['vertices'] <= 5 and obj['compactness'] > 1.2 and obj['compactness'] < 1.8) or
        is_triangular(obj['contour'])):
        return True
    
    return False

def is_square(obj):
    """Improved square detection using multiple metrics"""
    # Perfect square indicators
    if (obj['vertices'] == 4 and 
        0.7 <= obj['aspect_ratio'] <= 1.2 and 
        obj['solidity'] > 0.9 and
        obj['extent'] > 0.8):
        return True
    
    # Reasonable square indicators with partial evidence
    if ((obj['vertices'] >= 4 and obj['vertices'] <= 6) and 
        0.7 <= obj['aspect_ratio'] <= 1.3 and 
        obj['circularity'] < 0.87 and
        obj['extent'] > 0.75):
        return True
    
    return False

def is_blue_square(obj):
    """Improved blue square detection with relaxed criteria"""
    # Strong square indicators for blue objects
    if (obj['vertices'] == 4 and 
        0.7 <= obj['aspect_ratio'] <= 1.3 and 
        obj['solidity'] > 0.85):
        return True
    
    # Relaxed criteria for challenging lighting/perspective cases
    if ((obj['vertices'] >= 4 and obj['vertices'] <= 6) and 
        0.6 <= obj['aspect_ratio'] <= 1.4 and
        obj['circularity'] < 0.9 and
        obj['compactness'] < 1.4 and
        obj['extent'] > 0.7):
        return True
    
    return False

def is_circle(obj):
    """Improved circle detection using multiple metrics"""
    # Strong circle indicators
    if obj['circularity'] > 0.85:
        return True
    
    # # Reasonable circle indicators with partial evidence
    # if (obj['circularity'] > 0.8 and
    #     obj['rot_aspect_ratio'] > 0.9 and obj['rot_aspect_ratio'] < 1.1):
    #     return True
    
    return False

def is_triangular(contour):
    """Advanced triangle detection looking at angles and sides"""
    # If we have exactly 3 points, it's definitely a triangle
    if len(contour) == 3:
        return True
    
    # Try with convex hull
    hull = cv2.convexHull(contour)
    if len(hull) == 3:
        return True
    
    # Try multiple approximation factors for challenging cases
    peri = cv2.arcLength(contour, True)
    for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.1]:
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        if len(approx) == 3:
            # Verify this approximation makes sense by checking angles
            points = approx.reshape(-1, 2)
            if len(points) == 3:
                # Calculate sides
                a = np.linalg.norm(points[0] - points[1])
                b = np.linalg.norm(points[1] - points[2])
                c = np.linalg.norm(points[2] - points[0])
                
                # Check that all sides are significant
                min_side = min(a, b, c)
                max_side = max(a, b, c)
                if min_side > max_side * 0.1:  # No degenerate triangles
                    return True
    
    # If we haven't returned True by now, it's probably not a triangle
    return False

def is_yellow_square(obj):
    """Improved blue square detection with relaxed criteria"""
    # Strong square indicators for blue objects
    if (obj['vertices'] == 4 and 
        0.7 <= obj['aspect_ratio'] <= 1.4 and 
        obj['solidity'] > 0.85):
        return True
    
    # Relaxed criteria for challenging lighting/perspective cases
    if ((obj['vertices'] >= 4 and obj['vertices'] <= 6) and 
        0.6 <= obj['aspect_ratio'] <= 1.4 and
        obj['circularity'] < 0.9 and
        obj['compactness'] < 1.4 and
        obj['extent'] > 0.7):
        return True
    
    return False

def process_color_mask(mask, image, color_name, roi_offset=(0, 0)):
    """Process a color mask to detect shapes with improved feature extraction"""
    # Unpack ROI offset
    roi_x, roi_y = roi_offset
    
    # Find contours with simple approximation to get initial candidates
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    
    # Process contours
    if hierarchy is not None and hierarchy.shape[0] > 0:
        for i, contour in enumerate(contours):
            # Check if this is an external contour
            if hierarchy[0, i, 3] == -1:  # No parent, so it's external
                # Filter very small contours
                area = cv2.contourArea(contour)
                if area < 100:  # Lower threshold to catch small objects
                    continue
                
                # Get enhanced shape features
                shape_info = get_shape_features(contour)
                
                # Calculate center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Calculate global coordinates by adding ROI offset
                global_cx = cx + roi_x
                global_cy = cy + roi_y
                
                # Add object info
                objects.append({
                    'color': color_name,
                    'shape': shape_info['shape'],
                    'area': area,
                    'contour': contour,  # Store the original contour
                    'approx': shape_info['approx'],
                    'cx': cx,  # Local coordinate (within ROI)
                    'cy': cy,  # Local coordinate (within ROI)
                    'global_cx': global_cx,  # Global coordinate (whole image)
                    'global_cy': global_cy,  # Global coordinate (whole image)
                    'vertices': shape_info['vertices'],
                    'circularity': shape_info['circularity'],
                    'compactness': shape_info['compactness'],
                    'solidity': shape_info['solidity'],
                    'aspect_ratio': shape_info['aspect_ratio'],
                    'extent': shape_info.get('extent', 0),
                    'rot_aspect_ratio': shape_info.get('rot_aspect_ratio', 0),
                    'circle_similarity': shape_info.get('circle_similarity', 0),
                    'shape_confidence': shape_info.get('shape_confidence', 0),
                    'roi_x': roi_x,  # Store ROI offset
                    'roi_y': roi_y
                })
    
    # Sort objects by area (largest first) for more reliable detection
    objects.sort(key=lambda x: x['area'], reverse=True)
    
    return objects

def draw_contour(image, obj):
    """Draw the contour of the object with improved visualization based on shape"""
    # Set color based on the object's color
    if obj['color'] == "Blue":
        color_bgr = (255, 0, 0)  # Blue in BGR
    elif obj['color'] == "Yellow":
        color_bgr = (0, 0, 0)  # Yellow in BGR
    else:
        color_bgr = (0, 0, 255)  # Red in BGR
    
    # Draw shape-specific visualizations
    if obj['shape'] == 'circle':
        # For circles, use ellipse fitting or min enclosing circle
        if 'original_contour' in obj and len(obj['original_contour']) >= 5:
            # Try fitEllipse for better circle visualization
            try:
                ellipse = cv2.fitEllipse(obj['original_contour'])
                cv2.ellipse(image, ellipse, color_bgr, 2)
                
                # Add center dot
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                cv2.circle(image, center, 3, (0, 0, 255), -1)
                
                # Add "CIRCLE" label near the shape
                cv2.putText(image, "CIRCLE", 
                           (int(ellipse[0][0]) - 30, int(ellipse[0][1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            except:
                # Fallback to minEnclosingCircle
                (x, y), radius = cv2.minEnclosingCircle(obj['original_contour'])
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(image, center, radius, color_bgr, 2)
                cv2.circle(image, center, 3, (0, 255, 255), -1)
                
                # Add "CIRCLE" label
                cv2.putText(image, "CIRCLE", 
                           (center[0] - 30, center[1] - int(radius) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        else:
            # Use contour's center and area to calculate radius
            radius = int(np.sqrt(obj['area'] / np.pi))
            center = (obj['cx'], obj['cy'])
            cv2.circle(image, center, radius, color_bgr, 2)
            cv2.circle(image, center, 3, (0, 255, 255), -1)
            
            # Add "CIRCLE" label
            cv2.putText(image, "CIRCLE", 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
    
    elif obj['shape'] == 'triangle':
        # For triangles, highlight the vertices and draw the approximated shape
        cv2.drawContours(image, [obj['approx']], -1, color_bgr, 2)
        
        # Draw vertices as small circles
        for point in obj['approx']:
            pt = (point[0][0], point[0][1])
            cv2.circle(image, pt, 3, (0, 255, 255), -1)
        
        # Add "TRIANGLE" label
        cv2.putText(image, "TRIANGLE", 
                   (obj['cx'] - 35, obj['cy'] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                   
    elif obj['shape'] == 'square':
        # For squares, draw the contour and highlight corners
        cv2.drawContours(image, [obj['approx']], -1, color_bgr, 2)
        
        # Draw vertices as small circles
        for point in obj['approx']:
            pt = (point[0][0], point[0][1])
            cv2.circle(image, pt, 3, (0, 255, 255), -1)
        
        # Add "SQUARE" label
        cv2.putText(image, "SQUARE", 
                   (obj['cx'] - 30, obj['cy'] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                   
    else:
        # For other shapes, just draw the contour
        cv2.drawContours(image, [obj['approx']], -1, color_bgr, 2)
        
        # Add shape name label
        cv2.putText(image, obj['shape'].upper(), 
                   (obj['cx'] - 30, obj['cy'] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
    
    # Draw center point for all shapes
    cv2.circle(image, (obj['cx'], obj['cy']), 3, (0, 255, 0), -1)
    
def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect specific shapes and colors")
    parser.add_argument("--object", type=str, default="all", 
                        choices=["red_circle", "blue_triangle", "blue_square", "red_square", "all"],
                        help="Which object to detect")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file (if not using camera)")
    
    args = parser.parse_args()
    
    target_object = args.object
    
    print(f"Shape and Color Detection - Target: {target_object}")
    print("--------------------------------------")
    print("Press 'q' to quit")
    print("Press '1' for red circle")
    print("Press '2' for blue triangle")
    print("Press '3' for blue square")
    print("Press '4' for red square")
    print("Press '0' for all objects")
    
    # Try to use camera or image
    try:
        if args.image:
            # Process static image
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Could not read image from {args.image}")
                return
                
            # Process the static image
            result = detect_shapes_and_colors(image, target_object)
            
            # Show result
            cv2.imshow("Shape Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Process webcam feed
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return
                
            current_target = target_object
            
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process frame with current target
                result = detect_shapes_and_colors(frame, current_target)
                
                # Show result
                cv2.imshow("Shape Detection", result)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    current_target = "red_circle"
                    print(f"Target changed to: {current_target}")
                elif key == ord('2'):
                    current_target = "blue_triangle"
                    print(f"Target changed to: {current_target}")
                elif key == ord('3'):
                    current_target = "blue_square"
                    print(f"Target changed to: {current_target}")
                elif key == ord('4'):
                    current_target = "red_square"
                    print(f"Target changed to: {current_target}")
                elif key == ord('0'):
                    current_target = "all"
                    print(f"Target changed to: {current_target}")
                    
            # Release camera
            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()