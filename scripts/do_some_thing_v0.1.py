#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
# If you want to read CSVs in the future, uncomment:
# import pandas as pd
# from tf.transformations import quaternion_from_euler

def load_waypoints_from_csv(csv_path):
    """
    Placeholder for CSV loading.
    Expected columns: x, y, z, qx, qy, qz, qw
    Or: x, y, z, roll, pitch, yaw (then convert to quaternion).
    """
    # df = pd.read_csv(csv_path)
    # waypoints = []
    # for idx, row in df.iterrows():
    #     p = Pose()
    #     p.position.x = row['x']
    #     p.position.y = row['y']
    #     p.position.z = row['z']
    #     # if quaternion columns present:
    #     p.orientation.x = row['qx']
    #     p.orientation.y = row['qy']
    #     p.orientation.z = row['qz']
    #     p.orientation.w = row['qw']
    #     # else, if Euler angles:
    #     # q = quaternion_from_euler(row['roll'], row['pitch'], row['yaw'])
    #     # p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    #     waypoints.append(p)
    # return waypoints
    raise NotImplementedError("CSV loading not yet implemented")

def main():
    # Initialize MoveIt and ROS node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('smooth_csv_waypoints', anonymous=True)
    group = moveit_commander.MoveGroupCommander("manipulator")

    # Apply velocity/acceleration scaling for smoother built-in timing
    group.set_max_velocity_scaling_factor(0.5)
    group.set_max_acceleration_scaling_factor(0.5)

    # ─── Stage 1: Build waypoint list ───
    waypoints = []

    # 1a) Prepend current pose for a smooth start
    start_pose = group.get_current_pose().pose
    waypoints.append(copy.deepcopy(start_pose))

    # 1b) Hardcoded test waypoints (replace these 3 with your own)
    wp = Pose()
    wp.position.x = start_pose.position.x - 0.2
    wp.position.y = start_pose.position.y + 0.0
    wp.position.z = start_pose.position.z + 0.0
    wp.orientation = copy.deepcopy(start_pose.orientation)
    waypoints.append(copy.deepcopy(wp))

    wp.position.x = start_pose.position.x + 0.0
    wp.position.y = start_pose.position.y + 0.2
    wp.position.z = start_pose.position.z + 0.0
    waypoints.append(copy.deepcopy(wp))

    # wp.position.x = start_pose.position.x + 0.0
    # wp.position.y = start_pose.position.y + 0.0
    # wp.position.z = start_pose.position.z + 0.0
    # waypoints.append(copy.deepcopy(wp))

    # ─── Alternative: load from CSV (future) ───
    # csv_path = "/path/to/your/waypoints.csv"
    # waypoints = [start_pose] + load_waypoints_from_csv(csv_path)

    # ─── Stage 2: Compute Cartesian path ───
    eef_step = 0.2      # meter resolution of interpolation
    traj_plan, fraction = group.compute_cartesian_path(
        waypoints,
        eef_step
    )

    if fraction < 0.99:
        rospy.logwarn(
            "Only {:.1f}% of the Cartesian path was planned. "
            "Check waypoint reachability or adjust eef_step."
            .format(fraction * 100)
        )

    # ─── Stage 3: Time-parameterize the trajectory ───
    current_state = group.get_current_state()
    timed_plan = group.retime_trajectory(
        current_state,
        traj_plan,
        velocity_scaling_factor=0.3
    )

    # ─── Stage 4: Execute ───
    group.execute(timed_plan, wait=True)
    rospy.loginfo("Trajectory execution complete")

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass