#!/usr/bin/env python3
"""
Simplified script to move a UR robot along a trajectory using MoveIt.
This version specifically addresses the compute_cartesian_path signature error.

Usage:
  python3 ur_moveit_trajectory.py --csv path/to/trajectory.csv
"""

import sys
import time
import argparse
import numpy as np
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

class URMoveitTrajectory:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ur_moveit_trajectory', anonymous=True)
        
        # Initialize MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Set the group name
        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        
        # Set planning parameters
        self.move_group.set_planning_time(10.0)
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.move_group.set_goal_position_tolerance(0.005)
        self.move_group.set_goal_orientation_tolerance(0.04)
        
        # Print info about the robot
        planning_frame = self.move_group.get_planning_frame()
        eef_link = self.move_group.get_end_effector_link()
        group_names = self.robot.get_group_names()
        
        print("\nMoveIt Setup Information:")
        print(f"  Planning frame: {planning_frame}")
        print(f"  End effector link: {eef_link}")
        print(f"  Available planning groups: {group_names}")
        
        # Get current state
        current_pose = self.move_group.get_current_pose().pose
        print("\nCurrent Robot Pose:")
        print(f"  Position: [{current_pose.position.x:.4f}, {current_pose.position.y:.4f}, {current_pose.position.z:.4f}]")
        print(f"  Orientation: [{current_pose.orientation.x:.4f}, {current_pose.orientation.y:.4f}, "
              f"{current_pose.orientation.z:.4f}, {current_pose.orientation.w:.4f}]")
              
    def load_trajectory_from_csv(self, csv_file, downsample=1):
        """Load trajectory waypoints from a CSV file"""
        try:
            # Load the CSV file
            data = np.loadtxt(csv_file, delimiter=',')
            print(f"Loaded trajectory with {data.shape[0]} points from {csv_file}")
            
            # Check data shape
            if data.ndim != 2 or data.shape[1] < 3:
                print("Error: CSV must have at least 3 columns (x,y,z)")
                return None
            
            # Apply downsampling if requested
            if downsample > 1:
                data = data[::downsample]
                print(f"Downsampled to {data.shape[0]} points")
                
            # Create waypoints
            waypoints = []
            current_pose = self.move_group.get_current_pose().pose
            
            for i, row in enumerate(data):
                x, y, z = row[0], row[1], row[2]
                
                # Use current orientation for all waypoints
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                pose.orientation = current_pose.orientation
                
                waypoints.append(pose)
            
            return waypoints
            
        except Exception as e:
            print(f"Error loading trajectory: {str(e)}")
            return None
    
    def move_joint_by_joint(self, waypoints):
        """Alternative execution method using individual joint moves"""
        print(f"\nExecuting {len(waypoints)} waypoints individually")
        
        for i, waypoint in enumerate(waypoints):
            print(f"Moving to waypoint {i+1}/{len(waypoints)}")
            
            # Set position target
            self.move_group.set_pose_target(waypoint)
            success = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            if not success:
                print(f"Failed to reach waypoint {i+1}")
                continue
                
            # Small pause between waypoints
            rospy.sleep(0.1)
        
        print("Trajectory execution completed!")
        return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Move UR robot along a trajectory using MoveIt')
    parser.add_argument('--csv', required=True, help='Path to trajectory CSV file')
    parser.add_argument('--downsample', type=int, default=30, 
                        help='Downsample factor for trajectory points (default: 30)')
    parser.add_argument('--use_gripper_tcp', action='store_true',
                        help='Use gripper_tcp as end-effector link')
    
    args = parser.parse_args()
    
    try:
        # Initialize trajectory controller
        controller = URMoveitTrajectory()
        
        # Set end-effector link if specified
        if args.use_gripper_tcp:
            controller.move_group.set_end_effector_link("gripper_tcp")
            print("Using gripper_tcp as end-effector link")
        
        # Load trajectory
        waypoints = controller.load_trajectory_from_csv(
            args.csv, 
            downsample=args.downsample
        )
        
        if not waypoints:
            print("Failed to load waypoints from CSV")
            return 1
        
        # Execute trajectory using joint-by-joint method
        controller.move_joint_by_joint(waypoints)
        
    except rospy.ROSInterruptException:
        print("Execution interrupted!")
        return 1
    except KeyboardInterrupt:
        print("Execution stopped by user!")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())