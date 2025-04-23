#!/usr/bin/env python
"""
ROS1 node to stream a time-parameterized spline from CSV to a UR3e using servoL commands.

This node reads 'splineTraj.csv' (columns: x, y, z, xd, yd, zd),
reads a constant orientation [rx, ry, rz] from ROS params,
and publishes URScript 'servoL' messages at high rate to control the robot.

Usage:
  rosparam set /spline_stream_node/spline_csv /path/to/splineTraj.csv
  rosparam set /spline_stream_node/orientation [rx,ry,rz]
  rosparam set /spline_stream_node/dt 0.008
  rosparam set /spline_stream_node/lookahead_time 0.1
  rosparam set /spline_stream_node/gain 300
  rosrun your_package ur3e_spline_stream_node.py
"""
import rospy
import numpy as np
from std_msgs.msg import String

def main():
    rospy.init_node('spline_stream_node', anonymous=False)

    # Parameters
    spline_csv = rospy.get_param('~spline_csv', 'splineTraj.csv')
    orientation = rospy.get_param('~orientation', [0.0, 0.0, 0.0])
    dt = rospy.get_param('~dt', 0.008)
    lookahead_time = rospy.get_param('~lookahead_time', 0.1)
    gain = rospy.get_param('~gain', 300)
    script_topic = rospy.get_param('~script_topic', '/ur_driver/URScript')

    # Load spline data
    try:
        data = np.loadtxt(spline_csv, delimiter=',')
    except Exception as e:
        rospy.logerr(f"Failed to load CSV '{spline_csv}': {e}")
        return

    if data.ndim != 2 or data.shape[1] < 3:
        rospy.logerr("CSV must have at least 3 columns (x,y,z,...)")
        return

    # Prepare publisher
    pub = rospy.Publisher(script_topic, String, queue_size=1)
    rospy.loginfo(f"Publishing servoL commands to '{script_topic}' every {dt}s")

    # Wait for publisher connection
    rospy.sleep(1.0)

    rate = rospy.Rate(1.0/dt)
    count = 0
    total = data.shape[0]

    for row in data:
        if rospy.is_shutdown():
            break

        x, y, z = row[0], row[1], row[2]
        rx, ry, rz = orientation

        # Format URScript command
        cmd = ("servoL([{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}], "
               "t={:.4f}, lookahead_time={:.4f}, gain={})".format(
                   x, y, z, rx, ry, rz, dt, lookahead_time, gain))
        pub.publish(cmd)
        count += 1
        rospy.logdebug(f"Step {count}/{total}: {cmd}")
        rate.sleep()

    rospy.loginfo("Finished streaming spline trajectory.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
