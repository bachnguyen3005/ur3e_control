#!/usr/bin/env python3
"""
Script to control a UR3e robot to move between two specified points.
Fixes:
  - Reports X/Y in the teach‐pendant frame (X+, Y–) instead of MoveIt’s frame.
  - Internally plans in MoveIt’s frame (no unintended flips).
  - Falls back to small joint‐space steps if Cartesian planning fails.
  - Applies TCP Z‐offset for display so reported Z matches the pendant.
  - Prints MoveIt error codes when a plan or execution fails.
"""

import sys
import argparse
import rospy
import moveit_commander
import geometry_msgs.msg
import numpy as np
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes

# Tool length offset from flange to actual TCP (in meters)
TOOL_OFFSET_Z = 0.2185


# def flip_xy_to_moveit(pt):
#     """Convert a [x, y, z] from teach‐pendant frame → MoveIt frame."""
#     # Pendant: +X forward, –Y left. MoveIt: –X forward, +Y left.
#     return [-pt[0], -pt[1], pt[2]]


# def flip_xy_to_display(x, y):#!/usr/bin/env python3

#  return -x, -y

import sys
import argparse
import rospy
import moveit_commander
import geometry_msgs.msg
import numpy as np
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes

# Tool length offset from flange to actual TCP (in meters)
TOOL_OFFSET_Z = 0.2185


# def flip_xy_to_moveit(pt):
#     """Convert a [x, y, z] from teach‐pendant frame → MoveIt frame."""
#     # Pendant: +X forward, –Y left. MoveIt: –X forward, +Y left.
#     return [-pt[0], -pt[1], pt[2]]


# def flip_xy_to_display(x, y):
#     """Convert a single (x, y) from MoveIt frame → teach‐pendant frame."""
#     return -x, -y


class UR3ePointToPointControl:
    def __init__(self):
        rospy.init_node('ur3e_point_to_point', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot       = moveit_commander.RobotCommander()
        self.scene       = moveit_commander.PlanningSceneInterface()
        self.move_group  = moveit_commander.MoveGroupCommander("manipulator")

        # Give /joint_states etc. time to arrive
        rospy.sleep(2.0)

        # Increase planning robustness
        self.move_group.set_planning_time(10)
        self.move_group.set_num_planning_attempts(10)

        # Velocity / acceleration / tolerances
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.move_group.set_goal_position_tolerance(0.001)
        self.move_group.set_goal_orientation_tolerance(0.01)

        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            queue_size=20
        )

        print("Planning frame:",      self.move_group.get_planning_frame())
        print("End effector link:",  self.move_group.get_end_effector_link())
        print("Available groups:",    self.robot.get_group_names())

    def report_error(self, context):
        """Print the last MoveIt error code."""
        code = self.move_group.get_last_error_code().val
        print(f"⚠️ {context} error code = {code} ({MoveItErrorCodes._TYPE})")
        return code

    def get_moveit_current_pose(self):
        """Raw pose in MoveIt’s frame (flange TCP)."""
        return self.move_group.get_current_pose().pose

    def get_display_current_pose(self):
        """
        Pose for the user:
          - X,Y flipped to match teach pendant
          - Z offset applied so it matches tool tip
        """
        raw = self.get_moveit_current_pose()
        # dx, dy = flip_xy_to_display(raw.position.x, raw.position.y)
        disp = geometry_msgs.msg.Pose()
        disp.position.x = raw.position.x
        disp.position.y = raw.position.y
        disp.position.z = raw.position.z + TOOL_OFFSET_Z
        disp.orientation  = raw.orientation
        return disp

    def create_pose(self, position, orientation=None):
        """
        Build a MoveIt‐frame Pose from position=[x,y,z] and optional quaternion.
        """
        if orientation is None:
            cur = self.get_moveit_current_pose()
            orientation = [
                cur.orientation.x,
                cur.orientation.y,
                cur.orientation.z,
                cur.orientation.w
            ]
        p = geometry_msgs.msg.Pose()
        p.position.x = position[0]
        p.position.y = position[1]
        p.position.z = position[2]
        p.orientation.x = orientation[0]
        p.orientation.y = orientation[1]
        p.orientation.z = orientation[2]
        p.orientation.w = orientation[3]
        return p

    def move_to_pose(self, target_pose, wait=True):
        """Joint-space plan+execute to a single Pose."""
        x,y,z = target_pose.position.x, target_pose.position.y, target_pose.position.z
        print(f"→ Joint-space move to [x={x:.3f}, y={y:.3f}, z={z:.3f}]")
        self.move_group.set_pose_target(target_pose)
        ok = self.move_group.go(wait=wait)
        self.move_group.clear_pose_targets()
        if not ok:
            self.report_error("joint-space move")
        return ok or not wait

    def move_in_cartesian_path(self, waypoints, eef_step=0.01):
        """
        Cartesian-path planning & execution.
        Falls back to joint-space interpolation if fraction==0.
        """
        print(f"\nPlanning Cartesian through {len(waypoints)} waypoint(s)…")
        self.move_group.set_start_state_to_current_state()

        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints, eef_step, True
        )
        print(f"  → planned fraction = {fraction:.3f}")

        if fraction == 0.0:
            print("  ⚠️ fraction==0, falling back to joint-space interpolation")
            start = self.get_moveit_current_pose()
            pts   = [start] + waypoints
            for i in range(1, len(pts)):
                sp, ep = pts[i-1], pts[i]
                dist = np.linalg.norm([
                    ep.position.x - sp.position.x,
                    ep.position.y - sp.position.y,
                    ep.position.z - sp.position.z
                ])
                steps = max(int(np.ceil(dist / eef_step)), 1)
                for j in range(1, steps+1):
                    t = j/steps
                    xyz = [
                        sp.position.x + t*(ep.position.x-sp.position.x),
                        sp.position.y + t*(ep.position.y-sp.position.y),
                        sp.position.z + t*(ep.position.z-sp.position.z)
                    ]
                    pose = self.create_pose(xyz, [
                        sp.orientation.x,
                        sp.orientation.y,
                        sp.orientation.z,
                        sp.orientation.w
                    ])
                    if not self.move_to_pose(pose):
                        print("  → interpolation step failed")
                        return False
            return True

        if fraction < 1.0:
            print("  ⚠️ Partial path planned (<100%). Aborting.")
            return False

        print("  → Executing Cartesian path…")
        ok = self.move_group.execute(plan, wait=True)
        if not ok:
            self.report_error("Cartesian execution")
        return ok

    def move_between_points(self, start_pt, end_pt, cartesian=True):
        """
        Move from start_pt → end_pt. Both are in MoveIt frame.
        If start_pt is None, use the raw MoveIt current pose.
        """
        raw = self.get_moveit_current_pose()
        ori = [
            raw.orientation.x,
            raw.orientation.y,
            raw.orientation.z,
            raw.orientation.w
        ]
        # Fix: do NOT flip raw when using it as start
        if start_pt is None:
            start_pt = [
                -raw.position.x,
                -raw.position.y,
                raw.position.z
            ]
        start_pose = self.create_pose(start_pt, ori)
        end_pose   = self.create_pose(end_pt  , ori)

        print(f"\nPoint-to-point (MoveIt frame):")
        print(f"  Start: {start_pt}")
        print(f"  End:   {end_pt}")

        if not np.allclose([
            raw.position.x, raw.position.y, raw.position.z
        ], start_pt, atol=0.005):
            if not self.move_to_pose(start_pose):
                return False

        if cartesian:
            ok = self.move_in_cartesian_path([end_pose])
            if not ok:
                print("  ⚠️ Cartesian failed, falling back to joint-space final move")
                return self.move_to_pose(end_pose)
            return True
        else:
            return self.move_to_pose(end_pose)

    def move_through_waypoints(self, waypoints, cartesian=True):
        raw = self.get_moveit_current_pose()
        ori = [
            raw.orientation.x,
            raw.orientation.y,
            raw.orientation.z,
            raw.orientation.w
        ]
        poses = [self.create_pose(wp, ori) for wp in waypoints]

        print(f"\nWaypoints (MoveIt frame): {waypoints}")
        if not self.move_to_pose(poses[0]):
            return False
        if cartesian and len(poses)>1:
            return self.move_in_cartesian_path(poses[1:])
        for p in poses[1:]:
            if not self.move_to_pose(p):
                return False
        return True

    def shutdown(self):
        self.move_group.stop()
        moveit_commander.roscpp_shutdown()
        print("UR3e control shut down")


def parse_point(s):
    try:
        return [float(v) for v in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid point: {s}")


def main():
    parser = argparse.ArgumentParser(description='UR3e point-to-point control')
    parser.add_argument('--start',     type=parse_point,
                        help='x,y,z start (pendant frame; default=current)')
    parser.add_argument('--end',       type=parse_point, required=True,
                        help='x,y,z end (pendant frame)')
    parser.add_argument('--waypoints', type=parse_point, nargs='+',
                        help='intermediate x,y,z waypoints (pendant frame)')
    parser.add_argument('--joint',     action='store_true',
                        help='force joint-space planning')
    args = parser.parse_args()

    # # Convert user (pendant) → MoveIt frame
    # if args.start:
    #     args.start     = flip_xy_to_moveit(args.start)
    # args.end          = flip_xy_to_moveit(args.end)
    # if args.waypoints:
    #     args.waypoints = [flip_xy_to_moveit(wp) for wp in args.waypoints]

    ctrl = UR3ePointToPointControl()

    # Show user the current pose
    cur = ctrl.get_display_current_pose()
    print(f"\nCurrent (pendant frame) → X: {cur.position.x:.3f}, "
          f"Y: {cur.position.y:.3f}, Z: {cur.position.z:.3f}")

    # Execute motion
    if args.waypoints:
        ok = ctrl.move_through_waypoints(args.waypoints, not args.joint)
    else:
        ok = ctrl.move_between_points(args.start, args.end, not args.joint)

    # Report final
    fin = ctrl.get_display_current_pose()
    if ok:
        print(f"\nDone → X: {fin.position.x:.3f}, "
              f"Y: {fin.position.y:.3f}, Z: {fin.position.z:.3f}")
    else:
        print("\nMovement failed.")

    ctrl.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())



class UR3ePointToPointControl:
    def __init__(self):
        rospy.init_node('ur3e_point_to_point', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot       = moveit_commander.RobotCommander()
        self.scene       = moveit_commander.PlanningSceneInterface()
        self.move_group  = moveit_commander.MoveGroupCommander("manipulator")

        # Give /joint_states etc. time to arrive
        rospy.sleep(2.0)

        # Increase planning robustness
        self.move_group.set_planning_time(10)
        self.move_group.set_num_planning_attempts(10)

        # Velocity / acceleration / tolerances
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.move_group.set_goal_position_tolerance(0.001)
        self.move_group.set_goal_orientation_tolerance(0.01)

        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            queue_size=20
        )

        print("Planning frame:",      self.move_group.get_planning_frame())
        print("End effector link:",  self.move_group.get_end_effector_link())
        print("Available groups:",    self.robot.get_group_names())

    def report_error(self, context):
        """Print the last MoveIt error code."""
        code = self.move_group.get_last_error_code().val
        print(f"⚠️ {context} error code = {code} ({MoveItErrorCodes._TYPE})")
        return code

    def get_moveit_current_pose(self):
        """Raw pose in MoveIt’s frame (flange TCP)."""
        return self.move_group.get_current_pose().pose

    def get_display_current_pose(self):
        """
        Pose for the user:
          - X,Y flipped to match teach pendant
          - Z offset applied so it matches tool tip
        """
        raw = self.get_moveit_current_pose()
        # dx, dy = flip_xy_to_display(raw.position.x, raw.position.y)
        disp = geometry_msgs.msg.Pose()
        disp.position.x = dx
        disp.position.y = dy
        disp.position.z = raw.position.z + TOOL_OFFSET_Z
        disp.orientation  = raw.orientation
        return disp

    def create_pose(self, position, orientation=None):
        """
        Build a MoveIt‐frame Pose from position=[x,y,z] and optional quaternion.
        """
        if orientation is None:
            cur = self.get_moveit_current_pose()
            orientation = [
                cur.orientation.x,
                cur.orientation.y,
                cur.orientation.z,
                cur.orientation.w
            ]
        p = geometry_msgs.msg.Pose()
        p.position.x = position[0]
        p.position.y = position[1]
        p.position.z = position[2]
        p.orientation.x = orientation[0]
        p.orientation.y = orientation[1]
        p.orientation.z = orientation[2]
        p.orientation.w = orientation[3]
        return p

    def move_to_pose(self, target_pose, wait=True):
        """Joint-space plan+execute to a single Pose."""
        x,y,z = target_pose.position.x, target_pose.position.y, target_pose.position.z
        print(f"→ Joint-space move to [x={x:.3f}, y={y:.3f}, z={z:.3f}]")
        self.move_group.set_pose_target(target_pose)
        ok = self.move_group.go(wait=wait)
        self.move_group.clear_pose_targets()
        if not ok:
            self.report_error("joint-space move")
        return ok or not wait

    def move_in_cartesian_path(self, waypoints, eef_step=0.01):
        """
        Cartesian-path planning & execution.
        Falls back to joint-space interpolation if fraction==0.
        """
        print(f"\nPlanning Cartesian through {len(waypoints)} waypoint(s)…")
        self.move_group.set_start_state_to_current_state()

        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints, eef_step, True
        )
        print(f"  → planned fraction = {fraction:.3f}")

        if fraction == 0.0:
            print("  ⚠️ fraction==0, falling back to joint-space interpolation")
            start = self.get_moveit_current_pose()
            pts   = [start] + waypoints
            for i in range(1, len(pts)):
                sp, ep = pts[i-1], pts[i]
                dist = np.linalg.norm([
                    ep.position.x - sp.position.x,
                    ep.position.y - sp.position.y,
                    ep.position.z - sp.position.z
                ])
                steps = max(int(np.ceil(dist / eef_step)), 1)
                for j in range(1, steps+1):
                    t = j/steps
                    xyz = [
                        sp.position.x + t*(ep.position.x-sp.position.x),
                        sp.position.y + t*(ep.position.y-sp.position.y),
                        sp.position.z + t*(ep.position.z-sp.position.z)
                    ]
                    pose = self.create_pose(xyz, [
                        sp.orientation.x,
                        sp.orientation.y,
                        sp.orientation.z,
                        sp.orientation.w
                    ])
                    if not self.move_to_pose(pose):
                        print("  → interpolation step failed")
                        return False
            return True

        if fraction < 1.0:
            print("  ⚠️ Partial path planned (<100%). Aborting.")
            return False

        print("  → Executing Cartesian path…")
        ok = self.move_group.execute(plan, wait=True)
        if not ok:
            self.report_error("Cartesian execution")
        return ok

    def move_between_points(self, start_pt, end_pt, cartesian=True):
        """
        Move from start_pt → end_pt. Both are in MoveIt frame.
        If start_pt is None, use the raw MoveIt current pose.
        """
        raw = self.get_moveit_current_pose()
        ori = [
            raw.orientation.x,
            raw.orientation.y,
            raw.orientation.z,
            raw.orientation.w
        ]
        # Fix: do NOT flip raw when using it as start
        if start_pt is None:
            start_pt = [
                -raw.position.x,
                -raw.position.y,
                raw.position.z
            ]
        start_pose = self.create_pose(start_pt, ori)
        end_pose   = self.create_pose(end_pt  , ori)

        print(f"\nPoint-to-point (MoveIt frame):")
        print(f"  Start: {start_pt}")
        print(f"  End:   {end_pt}")

        if not np.allclose([
            raw.position.x, raw.position.y, raw.position.z
        ], start_pt, atol=0.005):
            if not self.move_to_pose(start_pose):
                return False

        if cartesian:
            ok = self.move_in_cartesian_path([end_pose])
            if not ok:
                print("  ⚠️ Cartesian failed, falling back to joint-space final move")
                return self.move_to_pose(end_pose)
            return True
        else:
            return self.move_to_pose(end_pose)

    def move_through_waypoints(self, waypoints, cartesian=True):
        raw = self.get_moveit_current_pose()
        ori = [
            raw.orientation.x,
            raw.orientation.y,
            raw.orientation.z,
            raw.orientation.w
        ]
        poses = [self.create_pose(wp, ori) for wp in waypoints]

        print(f"\nWaypoints (MoveIt frame): {waypoints}")
        if not self.move_to_pose(poses[0]):
            return False
        if cartesian and len(poses)>1:
            return self.move_in_cartesian_path(poses[1:])
        for p in poses[1:]:
            if not self.move_to_pose(p):
                return False
        return True

    def shutdown(self):
        self.move_group.stop()
        moveit_commander.roscpp_shutdown()
        print("UR3e control shut down")


def parse_point(s):
    try:
        return [float(v) for v in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid point: {s}")


def main():
    parser = argparse.ArgumentParser(description='UR3e point-to-point control')
    parser.add_argument('--start',     type=parse_point,
                        help='x,y,z start (pendant frame; default=current)')
    parser.add_argument('--end',       type=parse_point, required=True,
                        help='x,y,z end (pendant frame)')
    parser.add_argument('--waypoints', type=parse_point, nargs='+',
                        help='intermediate x,y,z waypoints (pendant frame)')
    parser.add_argument('--joint',     action='store_true',
                        help='force joint-space planning')
    args = parser.parse_args()

    # # Convert user (pendant) → MoveIt frame
    # if args.start:
    #     args.start     = flip_xy_to_moveit(args.start)
    # args.end          = flip_xy_to_moveit(args.end)
    # if args.waypoints:
    #     args.waypoints = [flip_xy_to_moveit(wp) for wp in args.waypoints]

    ctrl = UR3ePointToPointControl()

    # Show user the current pose
    cur = ctrl.get_display_current_pose()
    print(f"\nCurrent (pendant frame) → X: {cur.position.x:.3f}, "
          f"Y: {cur.position.y:.3f}, Z: {cur.position.z:.3f}")

    # Execute motion
    if args.waypoints:
        ok = ctrl.move_through_waypoints(args.waypoints, not args.joint)
    else:
        ok = ctrl.move_between_points(args.start, args.end, not args.joint)

    # Report final
    fin = ctrl.get_display_current_pose()
    if ok:
        print(f"\nDone → X: {fin.position.x:.3f}, "
              f"Y: {fin.position.y:.3f}, Z: {fin.position.z:.3f}")
    else:
        print("\nMovement failed.")

    ctrl.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())