#!/usr/bin/env python3
"""
ROS2 Launch File for Autonomous Vehicle Parallel Parking System
Launches the odometry node and main control node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Odometry Node
        Node(
            package='robot_parking',
            executable='odometry_node',
            name='odometry_node',
            output='screen',
            parameters=[]
        ),
        
        # Main Vehicle Controller Node
        Node(
            package='robot_parking',
            executable='main_node',
            name='main_node',
            output='screen',
            parameters=[]
        ),
    ])