#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import pandas as pd


def main():
    # Initialize MoveIt and ROS node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('smooth_csv_waypoints', anonymous=True)
    group = moveit_commander.MoveGroupCommander("manipulator")

    # Capture the current tool orientation
    start_pose = group.get_current_pose().pose
    tool_ori = start_pose.orientation

    # Apply velocity/acceleration scaling for smoother timing
    group.set_max_velocity_scaling_factor(0.5)
    group.set_max_acceleration_scaling_factor(0.1)

    # ─── Stage 1: Load positions and build Pose waypoints ───
    csv_path = "/home/dinh/catkin_ws/src/ur3e_control/scripts/waypointsMatrix_11.csv"
    df = pd.read_csv(csv_path)
    waypoints = []
    # Flip the x and y when using catersian waypoint!!!! Cuz robot read that
    for _, row in df.iterrows():
        p = Pose()
        p.position.x = -float(row['x'])
        p.position.y = -float(row['y'])
        p.position.z = float(row['z'])
        # Inject constant orientation to lock wrist
        p.orientation = copy.deepcopy(tool_ori)
        waypoints.append(p)

    print("DEBUG waypoints: ", waypoints)
    # ─── Stage 2: Compute Cartesian path ───
    eef_step = 0.1  # 1 cm interpolation
    traj_plan, fraction = group.compute_cartesian_path(
        waypoints,
        eef_step
    )
    if fraction < 0.99:
        rospy.logwarn(
            "Only {:.1f}% of the Cartesian path was planned. "
            "Check waypoint reachability or adjust eef_step.".format(fraction * 100)
        )

    # ─── Stage 3: Time-parameterize the trajectory ───
    current_state = group.get_current_state()
    timed_plan = group.retime_trajectory(
        current_state,
        traj_plan,
        velocity_scaling_factor=0.1
    )

    # ─── Stage 4: Execute ───
    group.execute(timed_plan, wait=True)
    rospy.loginfo("Trajectory execution complete")

    #moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass