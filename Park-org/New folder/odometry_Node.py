#!/usr/bin/env python3
"""
ROS2 Odometry Node for Encoder-based Autonomous Vehicle
Converts raw optical encoder pulse counts into cumulative distance and velocity.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Int32, Float32

from config import RobotConfig, TopicConfig


class OdometryNode(Node):
    """Odometry calculator processing hardware encoder ticks"""
    
    def __init__(self):
        super().__init__('odometry_node')
        
        self.config = RobotConfig()
        
        # State variables
        self.total_pulses = 0
        self.last_pulses = 0
        self.total_distance_cm = 0.0
        self.distance_offset = 0.0  # For trip distance reset
        
        # Velocity calculation
        self.last_time = self.get_clock().now()
        self.last_distance = 0.0
        self.current_velocity = 0.0
        
        # QoS Profile for real-time telemetry
        realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.pulse_sub = self.create_subscription(
            Int32, TopicConfig.ENCODER_TOPIC, self.pulse_callback, realtime_qos
        )
        self.reset_sub = self.create_subscription(
            Int32, TopicConfig.ODOMETRY_RESET_TOPIC, self.reset_callback, 10
        )
        
        # Publishers
        self.distance_pub = self.create_publisher(
            Float32, TopicConfig.ODOMETRY_DISTANCE_TOPIC, realtime_qos
        )
        self.trip_pub = self.create_publisher(
            Float32, TopicConfig.ODOMETRY_TRIP_TOPIC, realtime_qos
        )
        self.velocity_pub = self.create_publisher(
            Float32, TopicConfig.ODOMETRY_VELOCITY_TOPIC, realtime_qos
        )
        
        # Velocity calculation timer (20 Hz)
        self.velocity_timer = self.create_timer(0.05, self.calculate_velocity)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("Odometry Node Started")
        self.get_logger().info(f"  Wheel Diameter: {self.config.WHEEL_DIAMETER} cm")
        self.get_logger().info(f"  Pulses Per Rev: {self.config.PULSES_PER_REV}")
        self.get_logger().info(f"  Distance Per Pulse: {self.config.DISTANCE_PER_PULSE:.3f} cm")
        self.get_logger().info("=" * 50)

    def pulse_callback(self, msg: Int32):
        """Process incoming encoder pulse count"""
        current_pulses = msg.data
        delta_pulses = current_pulses - self.last_pulses
        self.last_pulses = current_pulses
        
        delta_distance = delta_pulses * self.config.DISTANCE_PER_PULSE
        self.total_distance_cm += abs(delta_distance)
        self.total_pulses = current_pulses
        
        # Publish distance metrics
        dist_msg = Float32()
        dist_msg.data = self.total_distance_cm
        self.distance_pub.publish(dist_msg)
        
        trip_msg = Float32()
        trip_msg.data = self.total_distance_cm - self.distance_offset
        self.trip_pub.publish(trip_msg)

    def calculate_velocity(self):
        """Calculate velocity (cm/s) at periodic 20Hz interval"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        if dt > 0:
            self.current_velocity = (self.total_distance_cm - self.last_distance) / dt
            
            vel_msg = Float32()
            vel_msg.data = self.current_velocity
            self.velocity_pub.publish(vel_msg)
            
            self.last_distance = self.total_distance_cm
            self.last_time = current_time

    def reset_callback(self, msg: Int32):
        """Reset trip distance offset"""
        self.distance_offset = self.total_distance_cm
        self.get_logger().info(f"Trip distance reset at {self.total_distance_cm:.1f} cm")


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()