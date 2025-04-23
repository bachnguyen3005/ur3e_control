#!/usr/bin/env python3
# python csv_ikcon_calculator.py --input waypointsMatrix.csv --output ur3e_joint_configs.csv
import numpy as np
import roboticstoolbox as rtb
from spatialmath.base import transl, rpy2tr
from math import pi
import csv
import argparse
import os

# Import the custom ikcon implementation
from ikcon import ikcon

def load_points_from_csv(filename):
    """
    Load points from a CSV file.
    
    Args:
        filename: Path to CSV file with x, y, z coordinates
    
    Returns:
        List of [x, y, z] points
    """
    points = []
    
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract x, y, z coordinates
                point = [
                    float(row['x']),
                    float(row['y']),
                    float(row['z'])
                ]
                points.append(point)
        
        print(f"Loaded {len(points)} points from {filename}")
        return points
    except Exception as e:
        print(f"Error loading points from {filename}: {e}")
        return []

def save_joint_configs_to_csv(joint_configs, filename):
    """
    Save joint configurations to a CSV file.
    
    Args:
        joint_configs: List of joint configurations (each with 6 joint values)
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['joint1_rad', 'joint2_rad', 'joint3_rad', 'joint4_rad', 'joint5_rad', 'joint6_rad', 
                        'joint1_deg', 'joint2_deg', 'joint3_deg', 'joint4_deg', 'joint5_deg', 'joint6_deg'])
        
        # Write joint values (both in radians and degrees)
        for config in joint_configs:
            # Convert to degrees
            config_deg = np.rad2deg(config)
            
            # Combine radians and degrees in one row
            row = list(config) + list(config_deg)
            writer.writerow(row)
    
    print(f"Saved {len(joint_configs)} joint configurations to {filename}")

def calculate_joint_configs(points, orientation, q_prefer):
    """
    Calculate joint configurations for a list of points.
    
    Args:
        points: List of [x, y, z] coordinates
        orientation: [roll, pitch, yaw] orientation in radians
        q_prefer: Preferred joint configuration (for initial seed)
    
    Returns:
        List of joint configurations
    """
    roll, pitch, yaw = orientation
    
    # Create the robot
    ur3 = rtb.models.UR3()
    
    # Initialize list to store joint configurations
    joint_configs = []
    
    # Use the preferred configuration as the initial seed
    q_current = q_prefer.copy()
    
    # Process each point
    for i, point in enumerate(points):
        x, y, z = point
        
        # Create target transformation matrix
        Tep = transl(-x, -y, z) @ rpy2tr(0, pi/2, pi/2)
        
        print(f"\nPoint {i+1}/{len(points)}: [{x:.5f}, {y:.5f}, {z:.5f}]")
        print(f"Target pose:\n{Tep}")
        
        # Solve IK using the ikcon function
        try:
            sol, err, flag, out = ikcon(ur3, Tep, q0=q_current)
            joint_configs.append(sol)
                
        except Exception as e:
            print(f"Exception during IK calculation: {e}")
            # If there's an exception, use the preferred configuration
            joint_configs.append(q_prefer)
    
    return joint_configs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate joint configurations for waypoints')
    parser.add_argument('--input', '-i', default='waypointsMatrix.csv', help='Input CSV file with waypoints')
    parser.add_argument('--output', '-o', default='ur3e_joint_configs.csv', help='Output CSV file for joint configurations')
    parser.add_argument('--roll', type=float, default=0.0, help='Roll angle (rad)')
    parser.add_argument('--pitch', type=float, default=pi/2, help='Pitch angle (rad)')
    parser.add_argument('--yaw', type=float, default=pi/2, help='Yaw angle (rad)')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    # Load points from CSV file
    points = load_points_from_csv(args.input)
    
    if not points:
        print("No valid points found. Exiting.")
        return
    
    # Initial joint configuration (same as in test_ikcon.py)
    q0 = np.deg2rad(np.array([-180, -90, 0, -90, 0, 0]))
    q_prefer = np.deg2rad(np.array([61.26, -81.48, -92.51, -91.86, 85.49, 6.96]))
    
    # Get orientation from command line arguments
    orientation = [args.roll, args.pitch, args.yaw]
    print(f"Using orientation (roll, pitch, yaw): {orientation}")
    
    # Calculate joint configurations for all points
    joint_configs = calculate_joint_configs(points, orientation, q_prefer)
    
    # Save joint configurations to CSV file
    save_joint_configs_to_csv(joint_configs, args.output)
    
    print(f"\nProcessed {len(points)} points. Results saved to {args.output}")

if __name__ == "__main__":
    main()