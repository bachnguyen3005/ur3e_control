#!/usr/bin/env python3
# python ur3e-joint-executor.py --input ur3e_joint_configs.csv
import sys
import csv
import rospy

import moveit_commander
import moveit_msgs.msg
import actionlib
import numpy as np
from math import pi
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import argparse

class UR3eJointExecutor:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ur3e_joint_executor', anonymous=True)
        
        # Initialize MoveIt Commander
        moveit_commander.roscpp_initialize(sys.argv)
        
        # Get robot description
        robot = moveit_commander.RobotCommander()
        
        # Get scene description
        scene = moveit_commander.PlanningSceneInterface()
        
        # Initialize the move group for the UR3e arm
        group_name = "manipulator"  # Standard name for UR robots
        move_group = moveit_commander.MoveGroupCommander(group_name)
        
        # Set parameters for real robot operation
        move_group.set_max_velocity_scaling_factor(0.2)  # 20% of max velocity for safety
        move_group.set_max_acceleration_scaling_factor(0.1)  # 10% of max acceleration for safety

        
        # Create a publisher to publish trajectory visualization
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                      moveit_msgs.msg.DisplayTrajectory,
                                                      queue_size=20)
        
        # Store the necessary objects
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher

        # Print information about the robot
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()
        
        print("============ Planning frame: %s" % planning_frame)
        print("============ End effector link: %s" % eef_link)
        print("============ Available Planning Groups:", robot.get_group_names())
        print("============ Robot state:", robot.get_current_state())
        
        # Setup action client for trajectory execution
        # For real UR3e, the controller name is typically different than in simulation
        controller_name = '/scaled_pos_joint_traj_controller/follow_joint_trajectory'
        rospy.loginfo(f"Connecting to action server: {controller_name}")
        self.trajectory_client = actionlib.SimpleActionClient(
            controller_name,
            FollowJointTrajectoryAction
        )
        
        # Wait for action server with longer timeout for real robot
        rospy.loginfo("Waiting for trajectory action server...")
        if not self.trajectory_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Failed to connect to the trajectory action server.")
            self.use_trajectory_client = False
            rospy.logwarn("Will use MoveIt execution instead, but this may not work correctly on the real robot.")
        else:
            rospy.loginfo("Connected to trajectory action server")
            self.use_trajectory_client = True
    
    def load_joint_configs_from_csv(self, filename):
        """
        Load joint configurations from a CSV file.
        
        Args:
            filename: Path to CSV file with joint configurations
            
        Returns:
            List of joint configurations (list of 6 joint values in radians)
        """
        joint_configs = []
        
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Extract joint values in radians
                    config = [
                        float(row['joint1_rad']),
                        float(row['joint2_rad']),
                        float(row['joint3_rad']),
                        float(row['joint4_rad']),
                        float(row['joint5_rad']),
                        float(row['joint6_rad'])
                    ]
                    joint_configs.append(config)
            
            rospy.loginfo(f"Loaded {len(joint_configs)} joint configurations from {filename}")
            return joint_configs
            
        except Exception as e:
            rospy.logerr(f"Error loading joint configurations: {e}")
            return []
    
    def go_to_joint_positions(self, joint_goal, velocity_scaling=0.2):
        """Move the robot to specified joint positions with safety velocity scaling."""
        # Set velocity scaling for this specific movement
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        
        # Set the joint goal positions
        self.move_group.go(joint_goal, wait=True)
        
        # Calling stop to ensure no residual movement
        self.move_group.stop()
        
        # For testing: Check if we reached the joint goal
        current_joints = self.move_group.get_current_joint_values()
        return self.all_close(joint_goal, current_joints, 0.01)
    
    def all_close(self, goal, actual, tolerance):
        """Check if the goal and actual positions are within tolerance."""
        if type(goal) is list:
            for i in range(len(goal)):
                if abs(actual[i] - goal[i]) > tolerance:
                    return False
        
        return True
    
    def get_current_joint_values(self):
        """Get the current joint values."""
        return self.move_group.get_current_joint_values()
    
    def execute_joint_sequence(self, joint_configs, velocity_scaling=0.3, pause_time=0.01):
        """
        Execute a sequence of joint configurations.
        
        Args:
            joint_configs: List of joint configurations (list of 6 joint values)
            velocity_scaling: Velocity scaling factor (0.0-1.0)
            pause_time: Time to pause between configurations (seconds)
            
        Returns:
            bool: Success status
        """
        if not joint_configs:
            rospy.logerr("No joint configurations to execute")
            return False
        
        rospy.loginfo(f"Executing sequence of {len(joint_configs)} joint configurations")
        
        for i, config in enumerate(joint_configs):
            rospy.loginfo(f"Moving to configuration {i+1}/{len(joint_configs)}")
            
            # Move to the joint configuration
            success = self.go_to_joint_positions(config, velocity_scaling)
            
            if not success:
                rospy.logwarn(f"Failed to reach configuration {i+1}")
                
            # Pause between configurations
            if i < len(joint_configs) - 1:  # Don't pause after the last one
                rospy.sleep(pause_time)
        
        rospy.loginfo("Joint sequence execution completed")
        return True

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Execute joint configurations from CSV file')
        parser.add_argument('--input', '-i', default='ur3e_joint_configs.csv', help='Input CSV file with joint configurations')
        parser.add_argument('--velocity', '-v', type=float, default=0.1, help='Velocity scaling factor (0.0-1.0)')
        parser.add_argument('--pause', '-p', type=float, default=0.5, help='Pause time between configurations (seconds)')
        parser.add_argument('--home', '-m', action='store_true', help='Move to home position before starting')
        args = parser.parse_args()
        
        # Initialize the controller
        controller = UR3eJointExecutor()
        rospy.sleep(2)  # Allow time for initialization
        
        # Print current robot state
        print("Current joint values:", controller.get_current_joint_values())
        
        # Load joint configurations from CSV
        joint_configs = controller.load_joint_configs_from_csv(args.input)
        
        if not joint_configs:
            print("No valid joint configurations loaded. Exiting.")
            return
        
        # Ask for confirmation before moving robot
        if args.home:
            input("Press Enter to move the robot to home position...")
            home = [pi/2, -pi/2, 0, -pi/2, 0, 0]  # Standard home position for UR3e
            controller.go_to_joint_positions(home, velocity_scaling=args.velocity)
            rospy.sleep(1)
        
        input(f"Press Enter to execute sequence of {len(joint_configs)} joint configurations...")
        
        # Execute the joint sequence
        controller.execute_joint_sequence(joint_configs, velocity_scaling=args.velocity, pause_time=args.pause)
        
        # Ask to return to home position
        if input("Return to home position? (y/n): ").lower() == 'y':
            print("Returning to home position...")
            home = [pi/2, -pi/2, 0, -pi/2, 0, 0]
            controller.go_to_joint_positions(home, velocity_scaling=args.velocity)
        
        print("Execution completed successfully!")
        
    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Shutdown MoveIt cleanly
        moveit_commander.roscpp_shutdown()
        print("MoveIt shutdown complete")

if __name__ == '__main__':
    main()