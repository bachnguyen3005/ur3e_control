import cv2
import numpy as np
from cv2 import aruco
import numpy as np
import roboticstoolbox as rtb
from spatialmath.base import transl, rpy2tr
from math import pi
from ikcon import ikcon

def map_pixel_to_robot_frame(image_path, pixel_x, pixel_y, camera_intrinsics, marker_size_mm):
    """
    Maps a pixel coordinate in the image to the robot base frame using aruco markers
    with known size for improved accuracy.
    
    Args:
        image_path (str): Path to the image containing aruco markers
        pixel_x (int): X coordinate of the pixel to map
        pixel_y (int): Y coordinate of the pixel to map
        camera_intrinsics (dict): Camera intrinsic parameters
        marker_size_mm (float): Size of the aruco marker in mm
        
    Returns:
        tuple: ((x, y) position in robot base frame (mm), 
               marker_positions_camera dictionary, 
               marker_rotations_camera dictionary)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Define the known positions of the aruco markers in robot base frame (mm)
    marker_positions_robot = {
        6: (250-61-10, -450+61+5),   # top left
        7: (-250+61+10, -450+61+5),  # top right
        8: (250-61-10, -150-61-5),   # lower left
        5: (-250+61+10, -150-61-5)   # lower right
    }
    
    # Extract camera parameters
    camera_matrix = np.array(camera_intrinsics['K']).reshape(3, 3)
    dist_coeffs = np.array(camera_intrinsics['D'])
    
    # Detect aruco markers
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Adjust dictionary as needed
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 4:
        raise ValueError("Could not detect all 4 aruco markers in the image")
    
    # Create dictionaries to store the detected marker poses
    marker_positions_camera = {}
    marker_rotations_camera = {}  # Store rotation vectors too
    
    # Process each detected marker
    # In newer OpenCV versions, we need to estimate pose for each marker individually
    for i, marker_id in enumerate(ids.flatten()):
        # Get the corners of this marker
        marker_corners = corners[i][0]
        
        # Create the 3D coordinates of the marker corners in the marker's local coordinate system
        # The marker is considered to be on the z=0 plane with dimensions marker_size_mm x marker_size_mm
        objPoints = np.array([
            [-marker_size_mm/2, marker_size_mm/2, 0],
            [marker_size_mm/2, marker_size_mm/2, 0],
            [marker_size_mm/2, -marker_size_mm/2, 0],
            [-marker_size_mm/2, -marker_size_mm/2, 0]
        ], dtype=np.float32)
        
        # Estimate the pose of this marker
        success, rvec, tvec = cv2.solvePnP(
            objPoints, 
            marker_corners, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            # Store the marker position (translation vector)
            marker_positions_camera[marker_id] = (tvec[0][0], tvec[1][0], tvec[2][0])
            # Store the rotation vector for frame axis visualization
            marker_rotations_camera[marker_id] = rvec
    
    # Create a mapping from camera coordinate space to robot base frame
    # We'll use a least-squares approach to find the transformation
    
    # Prepare points in both coordinate systems
    points_camera = []
    points_robot = []
    
    for marker_id in marker_positions_robot.keys():
        if marker_id in marker_positions_camera:
            # Use only x,y coordinates since we're mapping to 2D robot frame
            points_camera.append(marker_positions_camera[marker_id][:2])
            points_robot.append(marker_positions_robot[marker_id])
    
    if len(points_camera) < 2:
        raise ValueError("Not enough corresponding markers found between camera and robot frame")
    
    points_camera = np.array(points_camera, dtype=np.float32)
    points_robot = np.array(points_robot, dtype=np.float32)
    
    # Find the affine transformation
    # This handles translation, rotation, and scaling
    transform_matrix, _ = cv2.estimateAffine2D(points_camera, points_robot) 
    # Undistort the input pixel
    pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    undistorted_pixel = cv2.undistortPoints(
        pixel_point, 
        camera_matrix, 
        dist_coeffs, 
        None, 
        camera_matrix
    )[0][0]
    
    # Convert the pixel to a ray in camera coordinates
    # First, create normalized coordinates
    x_normalized = (undistorted_pixel[0] - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y_normalized = (undistorted_pixel[1] - camera_matrix[1, 2]) / camera_matrix[1, 1]
    
    # Now we need to find the intersection of this ray with the plane defined by the markers
    # For simplicity, we'll use a reference Z depth (average of the marker depths)
    avg_z = np.mean([pos[2] for pos in marker_positions_camera.values()])
    # print("Average depth: ", avg_z)
    
    # Compute the 3D point at the average Z depth
    point_camera_3d = np.array([
        x_normalized * avg_z,
        y_normalized * avg_z
    ])
    
    # Transform to robot coordinates using the affine transformation
    point_robot = transform_matrix @ np.append(point_camera_3d, 1)
    
    return (point_robot[0], point_robot[1]), marker_positions_camera, marker_rotations_camera

# Create a visualization function to help with debugging
def visualize_mapping(image_path, camera_intrinsics, marker_size_mm, grid_step=50):
    """
    Visualizes the mapping from image coordinates to robot base frame.
    
    Args:
        image_path (str): Path to the image containing aruco markers
        camera_intrinsics (dict): Camera intrinsic parameters
        marker_size_mm (float): Size of the aruco marker in mm
        grid_step (int): Step size for the visualization grid
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Detect aruco markers for visualization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Extract camera parameters
    camera_matrix = np.array(camera_intrinsics['K']).reshape(3, 3)
    dist_coeffs = np.array(camera_intrinsics['D'])
    
    # Draw detected markers
    if ids is not None:
        aruco.drawDetectedMarkers(vis_image, corners, ids)
        
        # Process markers individually to draw coordinate axes
        for i, marker_id in enumerate(ids.flatten()):
            # Get the corners of this marker
            marker_corners = corners[i][0]
            
            # Create the 3D coordinates of the marker corners
            objPoints = np.array([
                [-marker_size_mm/2, marker_size_mm/2, 0],
                [marker_size_mm/2, marker_size_mm/2, 0],
                [marker_size_mm/2, -marker_size_mm/2, 0],
                [-marker_size_mm/2, -marker_size_mm/2, 0]
            ], dtype=np.float32)
            
            # Estimate the pose of this marker
            success, rvec, tvec = cv2.solvePnP(
                objPoints, 
                marker_corners, 
                camera_matrix, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                # Draw coordinate axis for this marker
                # The axis length is half the marker size for better visualization
                axis_length = marker_size_mm / 2
                cv2.drawFrameAxes(vis_image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
                
                # Add marker ID with its position in robot frame
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                cv2.putText(
                    vis_image,
                    f"ID: {marker_id}",
                    (center_x, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
    
    # We'll collect marker positions and rotations directly rather than using map_pixel_to_robot_frame
    marker_positions = {}
    marker_rotations = {}
    
    # Draw a grid of points and their mappings
    height, width = image.shape[:2]
    
    results = []
    
    for y in range(0, height, grid_step):
        for x in range(0, width, grid_step):
            try:
                # Map the point to robot coordinates
                result = map_pixel_to_robot_frame(
                    image_path, x, y, camera_intrinsics, marker_size_mm
                )
                robot_x, robot_y = result[0]
                
                # Draw a circle at the pixel
                cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
                
                # Add text showing the robot coordinates
                cv2.putText(
                    vis_image, 
                    f"({robot_x:.0f},{robot_y:.0f})", 
                    (x + 5, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, 
                    (0, 0, 255), 
                    1
                )
                
                results.append((x, y, robot_x, robot_y))
                
            except Exception as e:
                print(f"Error mapping point ({x}, {y}): {e}")
    
    # Save the visualization without attempting to display it
    cv2.imwrite("mapping_visualization.jpg", vis_image)
    print(f"Visualization saved as 'mapping_visualization.jpg'")
    
    return results

# Main program
if __name__ == "__main__":
    # Create the robot
    ur3 = rtb.models.UR3()
    q_prefer = np.deg2rad(np.array([61.26, -81.48, -92.51, -91.86, 85.49, 6.96]))
    
    # Camera intrinsic parameters
    camera_intrinsics = {
        'height': 480,
        'width': 640,
        'distortion_model': 'plumb_bob',
        'D': [0.06117127, 0.1186219, -0.00319266, -0.00094209, -0.75616137],
        'K': [596.68849646, 0.0, 317.08346319,
              0.0, 596.0051831, 247.34662529,
              0.0, 0.0, 1.0],
        'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'P': [596.68849646, 0.0, 317.0835119, 0.0,
              0.0, 596.0051831, 247.34658383, 0.0,
              0.0, 0.0, 1.0, 0.0]
    }
    
    # Define your aruco marker size in mm
    marker_size_mm = 36  # Replace with your actual marker size
    
    image_path = "/home/dinh/catkin_ws/src/ur3e_control/calibration_images.jpg"
    
    # Example 1: Map a single point
    pixel_x = 390
    pixel_y = 217
    
    try:
        result = map_pixel_to_robot_frame(
            image_path, pixel_x, pixel_y, camera_intrinsics, marker_size_mm
        )
        robot_x, robot_y = result[0]
        Tep = transl(-robot_x*0.001, -robot_y*0.001, 0.257) @ rpy2tr(0, pi/2, pi/2)
        print(f"Pixel ({pixel_x}, {pixel_y}) maps to robot frame coordinates ({robot_x:.2f}, {robot_y:.2f}) mm")
        sol, err, flag, out = ikcon(ur3, Tep, q0=q_prefer)
        print(f"Joint solution (deg): {np.rad2deg(sol)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Create a visualization
    try:
        print("Generating visualization...")
        visualize_mapping(image_path, camera_intrinsics, marker_size_mm, grid_step=80)
    except Exception as e:
        print(f"Visualization error: {e}")