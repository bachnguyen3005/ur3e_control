#!/usr/bin/env python3

"""

Standalone Python script to stream a time-parameterized spline CSV to a UR3e robot

via the Universal Robots ROS1 hardware interface.
 
This script publishes URScript `servoL` commands to the

`/ur/ur_hardware_interface/script_command` topic, which the driver executes on the physical UR3e.
 
Usage:

  source /opt/ros/<distro>/setup.bash

  source ~/catkin_ws/devel/setup.bash

  python3 ur3e_spline_stream_ur_driver.py --csv splineTraj.csv --orientation 0.0 0.0 0.0 --dt 0.008 --lookahead 0.1 --gain 300 --topic /ur/ur_hardware_interface/script_command
 
Ensure the ROS driver is running (e.g. ur_robot_driver bringup).

"""

import rospy

from std_msgs.msg import String

import numpy as np

import argparse

import time
 
def parse_args():

    parser = argparse.ArgumentParser(

        description='Stream spline CSV to UR3e via URScript servoL using UR hardware interface')

    parser.add_argument('--csv', '-c', required=True,

                        help='Path to splineTraj.csv containing at least [x,y,z] columns')

    parser.add_argument('--orientation', '-o', type=float, nargs=3, default=[0.0,0.0,0.0],

                        metavar=('RX','RY','RZ'), help='Tool orientation in axis-angle')

    parser.add_argument('--dt', '-d', type=float, default=0.008,

                        help='Time step between servoL commands (s)')

    parser.add_argument('--lookahead', '-l', type=float, default=0.1,

                        help='Lookahead time for servoL (s)')

    parser.add_argument('--gain', '-g', type=float, default=300,

                        help='ServoL gain')

    parser.add_argument('--topic', '-t', default='/ur/ur_hardware_interface/script_command',

                        help='ROS topic for URScript commands')

    return parser.parse_args()
 
if __name__ == '__main__':

    args = parse_args()
 
    # Initialize ROS node and publisher

    rospy.init_node('spline_streamer', anonymous=False)

    pub = rospy.Publisher(args.topic, String, queue_size=1)

    rospy.loginfo(f"Streaming spline to URScript topic '{args.topic}' at {1/args.dt:.1f}Hz.")

    rospy.sleep(1.0)  # allow publisher to connect
 
    # Load spline data

    try:

        data = np.loadtxt(args.csv, delimiter=',')

    except Exception as e:

        rospy.logerr(f"Failed to load CSV '{args.csv}': {e}")

        exit(1)
 
    if data.ndim != 2 or data.shape[1] < 3:

        rospy.logerr("CSV must have at least 3 columns [x,y,z]")

        exit(1)

    positions = data[:, :3]
 
    # Unpack parameters

    rx, ry, rz = args.orientation

    dt = args.dt

    lookahead = args.lookahead

    gain = args.gain
 
    rate = rospy.Rate(1.0 / dt)

    total = positions.shape[0]

    count = 0
 
    # Stream each point

    for pos in positions:

        if rospy.is_shutdown():

            break

        x, y, z = pos.tolist()

        # Build URScript servoL command

        cmd = (f"servoL([{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"

               f" t={dt:.4f}, lookahead_time={lookahead:.4f}, gain={gain})")

        pub.publish(String(data=cmd))

        count += 1

        rospy.logdebug(f"Sent {count}/{total}: {cmd}")

        rate.sleep()
 
    rospy.loginfo("Finished streaming spline trajectory to UR3e.")

 