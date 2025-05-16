import numpy as np
 
# Define a simple structure for Cube
class Cube:
    def __init__(self, id, position, color):
        self.id = id
        self.position = np.array(position)
        self.color = color
 
# Function to run the Cube Pickup task
def run_cube_pickup():
    # Define the cube list
    cube_list = [
        Cube('cube1', [0.4, 0.2, 0], 'red'),
        Cube('cube2', [0.1, 0.5, 0], 'blue'),
        Cube('cube3', [0.3, 0.3, 0], 'yellow')
    ]
 
    # Define robot's starting pose
    start_pose = np.array([0, 0, 0])
 
    # Define obstacle position (e.g., a cylinder)
    obstacle_center = np.array([0.0, -0.3, 0.3])  # Example obstacle at [x, y, z]
 
    # Call function to plan the cube pickup order
    pickup_sequence = plan_cube_pickup_order(cube_list, start_pose, obstacle_center)
 
    # Display the result
    print("Pickup Order:")
    for i, pickup in enumerate(pickup_sequence):
        print(f"{i+1}: {pickup['id']} at ({pickup['position'][0]:.2f}, {pickup['position'][1]:.2f}, {pickup['position'][2]:.2f})")
 
# Function to plan the cube pickup order
def plan_cube_pickup_order(cube_list, start_pose, obstacle_center):
    current_pose = start_pose
    remaining_cubes = cube_list[:]
    pickup_sequence = []
 
    # Define color priority
    color_priority = ['yellow', 'red', 'blue']
 
    # Loop over color priority
    for color in color_priority:
        color_cubes = [cube for cube in remaining_cubes if cube.color == color]
        while color_cubes:
            # Find the next closest cube of the same color
            distances = [np.linalg.norm(current_pose[:2] - cube.position[:2]) for cube in color_cubes]
            closest_idx = np.argmin(distances)
            next_cube = color_cubes[closest_idx]
 
            # Check for collision and calculate the path to the cube
            waypoints_matrix = check_and_plan_path(current_pose, next_cube.position, obstacle_center)
 
            # Add to pickup sequence
            for wp in waypoints_matrix:
                pickup_sequence.append({'id': next_cube.id, 'position': wp, 'color': next_cube.color})
 
            # Update current pose
            current_pose = next_cube.position
 
            # Remove the collected cube from the remaining cubes list
            remaining_cubes = [cube for cube in remaining_cubes if cube.id != next_cube.id]
            color_cubes = [cube for cube in color_cubes if cube.id != next_cube.id]
 
    return pickup_sequence
 
# Function to check and plan path with collision avoidance
def check_and_plan_path(P0, P2, C):
    safe_margin = 0.03  # Extra clearance in meters
    projected_cylinder_radius = 0.03  # Obstacle (cylinder) radius in meters
    R = projected_cylinder_radius + safe_margin  # Safe radius around the obstacle
    z_constant = 0.3  # Fixed z value for all waypoints
 
    # Check for collision and generate waypoints
    if check_collision(P0[:2], P2[:2], C[:2], R):
        print('Collision detected. Computing trapezoidal tangent-based avoidance path.')
        num_steps = 50  # Total number of waypoints for the avoidance path
        path_points = tangent_path_trapezoid(P0[:2], P2[:2], C[:2], R, num_steps)  # N×2 matrix [x, y]
        waypoints_matrix = np.hstack([path_points, np.full((path_points.shape[0], 1), z_constant)])
    else:
        print('Path is clear: commanding direct target.')
        # Direct path consists of the start and target XY positions.
        direct_path = np.array([P0[:2], P2[:2]])  # 2×2 matrix [x, y]
        waypoints_matrix = np.hstack([direct_path, np.full((direct_path.shape[0], 1), z_constant)])  # 2×3 matrix
 
    return waypoints_matrix
 
# Simple Collision Checker
def check_collision(P0, P1, C, R):
    P0 = np.array(P0)
    P1 = np.array(P1)
    C = np.array(C)
 
    # Compute the projection factor t of C onto the line segment P0->P1.
    v = P1 - P0
    w = C - P0
    t = np.dot(w, v) / np.dot(v, v)
    t = max(0, min(1, t))  # Clamp t between 0 and 1
    closest_point = P0 + t * v
    distance = np.linalg.norm(C - closest_point)
    return distance < R
 
# Trapezoidal Tangent-Path Generator
def tangent_path_trapezoid(P0, P2, C, R, num_steps):
    theta0 = np.arctan2(P0[1] - C[1], P0[0] - C[0])
    theta2 = np.arctan2(P2[1] - C[1], P2[0] - C[0])
 
    # Determine candidate tangent angles for P0.
    d0 = np.linalg.norm(P0 - C)
    alpha0 = np.arccos(R / d0)
    candidate_angles0 = [theta0 + alpha0, theta0 - alpha0]
 
    # Determine candidate tangent angles for P2.
    d2 = np.linalg.norm(P2 - C)
    alpha2 = np.arccos(R / d2)
    candidate_angles2 = [theta2 + alpha2, theta2 - alpha2]
 
    # Select the candidate pair with the smallest angular difference.
    best_diff = float('inf')
    for i in range(2):
        for j in range(2):
            diff = abs((candidate_angles2[j] - candidate_angles0[i]) % (2 * np.pi) - np.pi)
            if diff < best_diff:
                best_diff = diff
                tangent_angle0 = candidate_angles0[i]
                tangent_angle2 = candidate_angles2[j]
 
    # Compute the two tangent points T1 and T3.
    T1 = C + R * np.array([np.cos(tangent_angle0), np.sin(tangent_angle0)])
    T3 = C + R * np.array([np.cos(tangent_angle2), np.sin(tangent_angle2)])
 
    # Divide the path into three segments: P0 -> T1, T1 -> T3, T3 -> P2.
    n1 = round(num_steps * 0.3)
    seg1 = np.column_stack([np.linspace(P0[0], T1[0], n1), np.linspace(P0[1], T1[1], n1)])
 
    n2 = round(num_steps * 0.4)
    seg2 = np.column_stack([np.linspace(T1[0], T3[0], n2), np.linspace(T1[1], T3[1], n2)])
 
    n3 = num_steps - (n1 + n2)
    seg3 = np.column_stack([np.linspace(T3[0], P2[0], n3), np.linspace(T3[1], P2[1], n3)])
 
    waypoints = np.vstack([seg1, seg2, seg3])
    return waypoints
 
# Wrap Angle to [-pi, pi]
def wrap_to_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
 
if __name__ == "__main__":
    run_cube_pickup()