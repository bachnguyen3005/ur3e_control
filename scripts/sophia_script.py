import numpy as np
 
def run_cube_pickup():
    # Define cube list
    cube_list = [
        {'id': 'cube1', 'position': np.array([0.4, 0.2, 0]), 'color': 'red'},
        {'id': 'cube2', 'position': np.array([0.1, 0.5, 0]), 'color': 'blue'},
        {'id': 'cube3', 'position': np.array([0.3, 0.3, 0]), 'color': 'green'}
    ]
 
    # Define robot's starting pose
    start_pose = np.array([0.0, 0.0, 0.0])
 
    # Define obstacle center
    obstacle_center = np.array([0.0, -0.3, 0.3])
 
    # Plan pickup sequence
    pickup_sequence = plan_cube_pickup_order(cube_list, start_pose, obstacle_center)
 
    # Display results
    print("Pickup Order:")
    for i, cube in enumerate(pickup_sequence, 1):
        pos = cube['position']
        print(f"{i}: {cube['id']} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
 
def plan_cube_pickup_order(cube_list, start_pose, obstacle_center):
    current_pose = start_pose.copy()
    remaining_cubes = cube_list.copy()
    pickup_sequence = []
 
    while remaining_cubes:
        distances = [np.linalg.norm(current_pose[:2] - cube['position'][:2]) for cube in remaining_cubes]
        closest_idx = np.argmin(distances)
        next_cube = remaining_cubes[closest_idx]
 
        waypoints = check_and_plan_path(current_pose, next_cube['position'], obstacle_center)
 
        for wp in waypoints:
            pickup_sequence.append({'id': next_cube['id'], 'position': wp, 'color': next_cube['color']})
 
        current_pose = next_cube['position']
        del remaining_cubes[closest_idx]
 
    return pickup_sequence
 
def check_and_plan_path(P0, P2, C):
    safe_margin = 0.03
    projected_cylinder_radius = 0.03
    R = projected_cylinder_radius + safe_margin
    z_constant = 0.3
 
    if check_collision(P0[:2], P2[:2], C[:2], R):
        print("Collision detected. Computing trapezoidal tangent-based avoidance path.")
        num_steps = 50
        path_xy = tangent_path_trapezoid(P0[:2], P2[:2], C[:2], R, num_steps)
        waypoints = [np.array([x, y, z_constant]) for x, y in path_xy]
    else:
        print("Path is clear: commanding direct target.")
        waypoints = [
            np.array([*P0[:2], z_constant]),
            np.array([*P2[:2], z_constant])
        ]
    return waypoints
 
def check_collision(P0, P1, C, R):
    P0, P1, C = np.array(P0), np.array(P1), np.array(C)
    v = P1 - P0
    w = C - P0
    t = np.dot(w, v) / np.dot(v, v)
    t = np.clip(t, 0, 1)
    closest_point = P0 + t * v
    distance = np.linalg.norm(C - closest_point)
    return distance < R
 
def tangent_path_trapezoid(P0, P2, C, R, num_steps):
    def angle_wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
 
    theta0 = np.arctan2(P0[1] - C[1], P0[0] - C[0])
    theta2 = np.arctan2(P2[1] - C[1], P2[0] - C[0])
 
    d0 = np.linalg.norm(P0 - C)
    d2 = np.linalg.norm(P2 - C)
    alpha0 = np.arccos(R / d0)
    alpha2 = np.arccos(R / d2)
 
    candidate_angles0 = [theta0 + alpha0, theta0 - alpha0]
    candidate_angles2 = [theta2 + alpha2, theta2 - alpha2]
 
    best_diff = float('inf')
    for angle0 in candidate_angles0:
        for angle2 in candidate_angles2:
            diff = abs(angle_wrap(angle2 - angle0))
            if diff < best_diff:
                best_diff = diff
                tangent_angle0, tangent_angle2 = angle0, angle2
 
    T1 = C + R * np.array([np.cos(tangent_angle0), np.sin(tangent_angle0)])
    T3 = C + R * np.array([np.cos(tangent_angle2), np.sin(tangent_angle2)])
 
    n1 = round(num_steps * 0.3)
    n2 = round(num_steps * 0.4)
    n3 = num_steps - n1 - n2
 
    seg1 = np.linspace(P0, T1, n1)
    seg2 = np.linspace(T1, T3, n2)
    seg3 = np.linspace(T3, P2, n3)
 
    return np.vstack((seg1, seg2, seg3))
 
# Run the simulation
if __name__ == "__main__":
    run_cube_pickup()