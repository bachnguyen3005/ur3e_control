#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import geometry_msgs.msg

def main():
    # Initialise the ROS node and MoveIt! commander
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur3e_waypoints', anonymous=True)

    # Interfaces to robot and planning scene
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    # Move group for the UR3e
    group = moveit_commander.MoveGroupCommander("manipulator")

    # Optional: scale down speed/acceleration for safety
    group.set_max_velocity_scaling_factor(0.2)
    group.set_max_acceleration_scaling_factor(0.2)

    # Get current pose to start from
    start_pose = group.get_current_pose().pose
    print("Start pose: ", start_pose)

    # Define waypoints list
    waypoints = []

    # First waypoint: offset +0.1 in x, +0.1 in y, +0.2 in z
    wpose = copy.deepcopy(start_pose)
    wpose.position.x += 0.1
    wpose.position.y += 0.0
    wpose.position.z += 0.0
    waypoints.append(copy.deepcopy(wpose))

    # # Second waypoint: further +0.2 in x, -0.1 in y
    # wpose.position.x += 0.2
    # wpose.position.y -= 0.1
    # waypoints.append(copy.deepcopy(wpose))

    # # Third waypoint: -0.1 in x, -0.1 in y, -0.1 in z
    # wpose.position.x -= 0.1
    # wpose.position.y -= 0.1
    # wpose.position.z -= 0.1
    # waypoints.append(copy.deepcopy(wpose))

    # Compute a Cartesian path connecting the waypoints smoothly
    # eef_step: resolution of trajectory (meters)
    # jump_threshold: disable jump_threshold checking
    (plan, fraction) = group.compute_cartesian_path(
        waypoints,
        0.01, 
        True
        
    )

    # Execute the trajectory
    if fraction == 1.0:
        rospy.loginfo("Full path planned (100%). Executing...")
        group.execute(plan, wait=True)
        rospy.loginfo("Motion complete.")
    else:
        rospy.logwarn("Only %.2f%% of path planned. Adjust waypoints or parameters." % (fraction*100.0))

    # Shutdown MoveIt cleanly
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass