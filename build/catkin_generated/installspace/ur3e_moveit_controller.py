#!/usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler
import actionlib
import numpy as np
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

class UR3eController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ur3e_moveit_controller', anonymous=True)
        
        # Initialize MoveIt Commander
        moveit_commander.roscpp_initialize(sys.argv)
        
        # Get robot description
        robot = moveit_commander.RobotCommander()
        
        # Get scene description
        scene = moveit_commander.PlanningSceneInterface()
        
        # Initialize the move group for the UR3e arm
        group_name = "manipulator"  # This should match your MoveIt config
        move_group = moveit_commander.MoveGroupCommander(group_name)
        
        # Create a publisher to publish trajectory visualization
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                      moveit_msgs.msg.DisplayTrajectory,
                                                      queue_size=20)
        
        # Store the necessary objects
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher

        # Print some basic information about the robot
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()
        
        print("============ Planning frame: %s" % planning_frame)
        print("============ End effector link: %s" % eef_link)
        print("============ Available Planning Groups:", robot.get_group_names())
        print("============ Robot state:", robot.get_current_state())
        
        # Setup action client for trajectory execution (optional, for more control)
        # The controller name might differ based on your setup
        controller_name = '/scaled_pos_joint_traj_controller/follow_joint_trajectory'
        rospy.loginfo(f"Connecting to action server: {controller_name}")
        self.trajectory_client = actionlib.SimpleActionClient(
            controller_name,
            FollowJointTrajectoryAction
        )
        # Use a shorter timeout to avoid long wait if the controller doesn't exist
        if not self.trajectory_client.wait_for_server(rospy.Duration(2.0)):
            rospy.logwarn("Failed to connect to the trajectory action server. Will use MoveIt execution instead.")
            self.use_trajectory_client = False
        else:
            rospy.loginfo("Connected to trajectory action server")
            self.use_trajectory_client = True
            
    def go_to_joint_positions(self, joint_goal):
        """Move the robot to specified joint positions."""
        # Set the joint goal positions
        self.move_group.go(joint_goal, wait=True)
        
        # Calling stop to ensure no residual movement
        self.move_group.stop()
        
        # For testing: Check if we reached the joint goal
        current_joints = self.move_group.get_current_joint_values()
        return self.all_close(joint_goal, current_joints, 0.01)
    
    def go_to_pose_goal(self, pose_goal):
        """Move the end-effector to a specified pose."""
        # Set the pose target
        self.move_group.set_pose_target(pose_goal)
        
        # Plan and execute the motion
        plan = self.move_group.go(wait=True)
        
        # Calling stop to ensure no residual movement
        self.move_group.stop()
        
        # Clear any pose targets
        self.move_group.clear_pose_targets()
        
        # For testing: Check if we reached the pose goal
        current_pose = self.move_group.get_current_pose().pose
        return self.all_close(pose_goal, current_pose, 0.01)
    
    def plan_cartesian_path(self, waypoints, eef_step=0.01, jump_threshold=0.0):
        """Plan a Cartesian path through the specified waypoints."""
        # Plan the Cartesian path
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,      # waypoints to follow
            eef_step,       # eef_step
            jump_threshold, # jump_threshold
            True            # avoid_collisions
        )
        
        return plan, fraction
    
    def display_trajectory(self, plan):
        """Display the planned trajectory in RViz."""
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        
        # Publish the trajectory
        self.display_trajectory_publisher.publish(display_trajectory)
    
    def execute_plan(self, plan):
        """Execute a previously computed plan."""
        return self.move_group.execute(plan, wait=True)
    
    def all_close(self, goal, actual, tolerance):
        """Check if the goal and actual positions are within tolerance."""
        if type(goal) is list:
            for i in range(len(goal)):
                if abs(actual[i] - goal[i]) > tolerance:
                    return False
        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_close(goal.pose, actual, tolerance)
        elif type(goal) is geometry_msgs.msg.Pose:
            return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)
        
        return True
    
    def execute_trajectory_directly(self, plan):
        """Execute the trajectory directly using the action client."""
        if not hasattr(self, 'use_trajectory_client') or not self.use_trajectory_client:
            rospy.logwarn("Action client not available, using MoveIt execution instead")
            return self.execute_plan(plan)
            
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = plan.joint_trajectory
        
        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result()
        
        return self.trajectory_client.get_result()
    
    def add_box(self, name, pose, size=(0.1, 0.1, 0.1)):
        """Add a box to the planning scene."""
        self.scene.add_box(name, pose, size)
        
def main():
    try:
        # Initialize the controller
        controller = UR3eController()
        rospy.sleep(1)  # Allow time for initialization
        
        # Example 1: Move to a joint configuration
        print("Moving to joint configuration...")
        home = [0, -pi/2, 0, -pi/2, 0, 0]  # Example joint values in radians
        controller.go_to_joint_positions(home)
        rospy.sleep(1)
        
        # Example 2: Move to a pose
        print("Moving to next joint configuration...")
        arrow_shape = [0, -60, 120, -150, -90, 0]  # Example joint values in radians
        controller.go_to_joint_positions(np.deg2rad(arrow_shape))
        rospy.sleep(1)
       
    except rospy.ROSInterruptException:
        print("Program terminated!")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Shutdown MoveIt cleanly
        moveit_commander.roscpp_shutdown()
        print("MoveIt shutdown complete")

if __name__ == '__main__':
    main()