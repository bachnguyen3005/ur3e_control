#!/usr/bin/env python3
"""
Simple script to control a UR3e robot to move along the X axis.

This script provides a standalone function to move the UR3e end-effector
along the X axis by a specified distance.

Usage:
  python3 ur3e_x_axis_control.py --distance 0.1
"""

import sys
import argparse
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from moveit_msgs.msg import DisplayTrajectory
from std_msgs.msg import Bool


class UR3eAxisControl:
    """Controller class for moving UR3e robot along axes."""

    def __init__(self):
        """Initialize the robot connection and MoveIt interfaces."""
        # Initialize ROS node
        rospy.init_node('ur3e_axis_control', anonymous=True)
        
        # Initialize MoveIt Commander
        moveit_commander.roscpp_initialize(sys.argv)
        
        # Robot interface
        self.robot = moveit_commander.RobotCommander()
        
        # Planning scene interface
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Move group for the manipulator
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")
        
        # Set planning parameters
        self.move_group.set_max_velocity_scaling_factor(0.3)  # 30% of max velocity
        self.move_group.set_max_acceleration_scaling_factor(0.3)  # 30% of max acceleration
        self.move_group.set_goal_position_tolerance(0.001)  # 1mm
        self.move_group.set_goal_orientation_tolerance(0.01)  # ~0.6 degrees
        
        # Publisher for trajectory visualization
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            queue_size=20)
        
        # Print robot info
        planning_frame = self.move_group.get_planning_frame()
        eef_link = self.move_group.get_end_effector_link()
        group_names = self.robot.get_group_names()
        
        print(f"Planning frame: {planning_frame}")
        print(f"End effector link: {eef_link}")
        print(f"Available planning groups: {group_names}")

    def move_along_x_axis(self, distance):
        """
        Move the robot end-effector along the X axis by the specified distance.
        
        Args:
            distance (float): Distance in meters to move along X axis.
                              Positive = forward, Negative = backward.
        
        Returns:
            bool: True if movement was successful, False otherwise.
        """
        print(f"Moving along X axis by {distance} meters...")
        
        # Get current pose
        current_pose = self.move_group.get_current_pose().pose
        print(f"Current position: [{current_pose.position.x:.4f}, "
              f"{current_pose.position.y:.4f}, {current_pose.position.z:.4f}]")
        
        # Create target pose (same as current but with modified X)
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = current_pose.position.x + distance
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z
        target_pose.orientation = current_pose.orientation
        
        print(f"Target position: [{target_pose.position.x:.4f}, "
              f"{target_pose.position.y:.4f}, {target_pose.position.z:.4f}]")
        
        # Set the target pose
        self.move_group.set_pose_target(target_pose)
        
        # Plan and execute
        plan_success, plan, planning_time, error_code = self.move_group.plan()
        
        if not plan_success:
            print(f"Planning failed with error code: {error_code}")
            self.move_group.clear_pose_targets()
            return False
        
        # Execute the plan
        execution_success = self.move_group.execute(plan, wait=True)
        
        # Clear targets from MoveIt
        self.move_group.clear_pose_targets()
        self.move_group.stop()
        
        if execution_success:
            print("X-axis movement completed successfully")
            new_pose = self.move_group.get_current_pose().pose
            print(f"New position: [{new_pose.position.x:.4f}, "
                  f"{new_pose.position.y:.4f}, {new_pose.position.z:.4f}]")
            return True
        else:
            print("X-axis movement failed during execution")
            return False
    
    def move_along_axis(self, axis, delta):
        """
        Move the robot end-effector along the specified axis by delta amount.
        
        Args:
            axis (str): The axis to move along ('x', 'y', or 'z')
            delta (float): Distance in meters to move along the axis
        
        Returns:
            bool: True if movement was successful, False otherwise
        """
        if axis.lower() not in ['x', 'y', 'z']:
            print(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
            return False
        
        print(f"Moving along {axis.upper()} axis by {delta} meters...")
        
        # Get current pose
        current_pose = self.move_group.get_current_pose().pose
        
        # Create target pose (same as current but with modified axis)
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = current_pose.position.x
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z
        
        # Update the appropriate axis
        if axis.lower() == 'x':
            target_pose.position.x += delta
        elif axis.lower() == 'y':
            target_pose.position.y += delta
        elif axis.lower() == 'z':
            target_pose.position.z += delta
            
        # Keep the same orientation
        target_pose.orientation = current_pose.orientation
        
        # Set the target pose
        self.move_group.set_pose_target(target_pose)
        
        # Plan and execute
        plan_success, plan, planning_time, error_code = self.move_group.plan()
        
        if not plan_success:
            print(f"Planning failed with error code: {error_code}")
            self.move_group.clear_pose_targets()
            return False
        
        # Execute the plan
        execution_success = self.move_group.execute(plan, wait=True)
        
        # Clear targets from MoveIt
        self.move_group.clear_pose_targets()
        self.move_group.stop()
        
        if execution_success:
            print(f"{axis.upper()}-axis movement completed successfully")
            return True
        else:
            print(f"{axis.upper()}-axis movement failed during execution")
            return False

    def shutdown(self):
        """Clean up and shut down the robot connection."""
        moveit_commander.roscpp_shutdown()
        print("UR3e control shut down")


def main():
    """Main function to parse arguments and control the robot."""
    parser = argparse.ArgumentParser(description='Control UR3e robot along X axis')
    parser.add_argument('--distance', type=float, default=0.1,
                        help='Distance to move along X axis in meters (default: 0.1)')
    parser.add_argument('--axis', type=str, default='x',
                        help='Axis to move along (x, y, or z) (default: x)')
    args = parser.parse_args()
    
    try:
        controller = UR3eAxisControl()
        
        # Check current robot position
        current_pose = controller.move_group.get_current_pose().pose
        print("Current robot position:")
        print(f"  X: {current_pose.position.x:.4f}")
        print(f"  Y: {current_pose.position.y:.4f}")
        print(f"  Z: {current_pose.position.z:.4f}")
        
        # Move the robot
        success = controller.move_along_axis(args.axis, args.distance)
        
        if success:
            new_pose = controller.move_group.get_current_pose().pose
            print("\nMovement complete. New position:")
            print(f"  X: {new_pose.position.x:.4f}")
            print(f"  Y: {new_pose.position.y:.4f}")
            print(f"  Z: {new_pose.position.z:.4f}")
            print(f"\nMoved {args.distance:.4f}m along {args.axis.upper()} axis")
        else:
            print("\nMovement failed")
            
        # Clean up
        controller.shutdown()
        
    except rospy.ROSInterruptException:
        print("Program interrupted")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == '__main__':
    sys.exit(main())