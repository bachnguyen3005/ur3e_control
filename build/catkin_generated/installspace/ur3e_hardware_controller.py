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

class UR3eRealRobotController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ur3e_real_robot_controller', anonymous=True)
        
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
    
    def go_to_pose_goal(self, pose_goal, velocity_scaling=0.2):
        """Move the end-effector to a specified pose with safety velocity scaling."""
        # Set velocity scaling for this specific movement
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        
        # Set the pose target
        self.move_group.set_pose_target(pose_goal)
        
        # Plan and execute the motion
        success = self.move_group.go(wait=True)
        
        # Calling stop to ensure no residual movement
        self.move_group.stop()
        
        # Clear any pose targets
        self.move_group.clear_pose_targets()
        
        # For testing: Check if we reached the pose goal
        current_pose = self.move_group.get_current_pose().pose
        return success and self.all_close(pose_goal, current_pose, 0.01)
    
    def plan_cartesian_path(self, waypoints, eef_step=0.01, jump_threshold=0.0):
        """Plan a Cartesian path through the specified waypoints."""
        # Plan the Cartesian path
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,      # waypoints to follow
            eef_step,       # eef_step
            jump_threshold, # jump_threshold
            True            # avoid_collisions
        )
        
        if fraction < 0.9:
            rospy.logwarn(f"Could only compute {fraction:.2%} of the requested path!")
        
        return plan, fraction
    
    def display_trajectory(self, plan):
        """Display the planned trajectory in RViz."""
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        
        # Publish the trajectory
        self.display_trajectory_publisher.publish(display_trajectory)
        rospy.loginfo("Trajectory displayed in RViz")
    
    def execute_plan(self, plan):
        """Execute a previously computed plan."""
        if self.use_trajectory_client:
            return self.execute_trajectory_directly(plan)
        else:
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
        
        rospy.loginfo("Sending trajectory to controller...")
        self.trajectory_client.send_goal(goal)
        
        # Monitor execution with feedback
        rospy.loginfo("Waiting for trajectory execution...")
        self.trajectory_client.wait_for_result()
        
        result = self.trajectory_client.get_result()
        if result:
            rospy.loginfo("Trajectory execution completed")
        else:
            rospy.logwarn("Trajectory execution may have failed")
        
        return result
    
    def add_collision_object(self, name, pose, size=(0.1, 0.1, 0.1)):
        """Add a collision object to the planning scene."""
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.move_group.get_planning_frame()
        box_pose.pose = pose
        self.scene.add_box(name, box_pose, size)
        rospy.loginfo(f"Added collision object: {name}")
        
    def remove_collision_object(self, name):
        """Remove a collision object from the planning scene."""
        self.scene.remove_world_object(name)
        rospy.loginfo(f"Removed collision object: {name}")
        
    def get_current_pose(self):
        """Get the current pose of the end effector."""
        return self.move_group.get_current_pose().pose
        
    def get_current_joint_values(self):
        """Get the current joint values."""
        return self.move_group.get_current_joint_values()

def main():
    try:
        # Initialize the controller
        controller = UR3eRealRobotController()
        rospy.sleep(2)  # Allow time for initialization and controller connection
        
        # Print current robot state
        print("Current joint values:", controller.get_current_joint_values())
        print("Current end effector pose:", controller.get_current_pose())
        
        # Ask for confirmation before moving the real robot
        input("Press Enter to move the robot to home position...")
        
        # Example 1: Move to home position with low velocity (10%)
        print("Moving to home position...")
        home = [pi/2, -pi/2, 0, -pi/2, 0, 0]  # Standard home position for UR3e
        controller.go_to_joint_positions(home, velocity_scaling=0.1)
        rospy.sleep(1)
        
        # Ask for confirmation before next movement
        input("Press Enter to move to arrow shape position...")
        
        # Example 2: Move to a specific joint configuration
        print("Moving to arrow shape position...")
        arrow_shape = [90, -60, 120, -150, -90, 0]  # Joint values in degrees
        controller.go_to_joint_positions(np.deg2rad(arrow_shape), velocity_scaling=0.1)
        rospy.sleep(1)
        
        input("Press Enter to move to L shape position...")
        print("Moving to L-shape position...")
        L_shape = [90, -90, 90, -90, -90, 0]  # Standard home position for UR3e
        controller.go_to_joint_positions(np.deg2rad(L_shape), velocity_scaling=0.1)
        rospy.sleep(1)
        
        # Ask for confirmation before returning to home
        input("Press Enter to return to home position...")
        
        # Return to home position
        print("Returning to home position...")
        controller.go_to_joint_positions(home, velocity_scaling=0.1)
        
        print("Motion sequence completed successfully!")
        
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