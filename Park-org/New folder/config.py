#!/usr/bin/env python3
"""
Configuration Parameters for Autonomous Parallel Parking Robot
Supports both AVIS Engine Simulator and ROS2 Physical Hardware setups.
"""

import math


class RobotConfig:
    """Robot physical dimensions and kinematic parameters"""
    ROBOT_LENGTH = 400.0        # Vehicle length (cm)
    ROBOT_WIDTH = 180.0         # Vehicle width (cm)
    WHEEL_DIAMETER = 60.0       # Wheel diameter (cm)
    WHEELBASE = 250.0           # Distance between front and rear axles (cm)
    
    # Encoder specification (Physical robot)
    PULSES_PER_REV = 16
    WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER
    DISTANCE_PER_PULSE = WHEEL_CIRCUMFERENCE / PULSES_PER_REV
    
    # Steering limits
    MAX_STEERING_ANGLE = 30     # Maximum steering angle (degrees)


class SensorConfig:
    """Sensor thresholds and sensor indices"""
    # Distance thresholds (cm)
    OBSTACLE_THRESHOLD = 200.0        # Distance below which an object is considered an obstacle
    MIN_SAFE_DISTANCE = 50.0          # Minimum safe distance from obstacles
    MAX_SENSOR_RANGE = 1000.0         # Maximum sensor operational range
    SPACE_START_THRESHOLD = 300.0     # Sensor reading above which an open spot is detected
    
    # Sensor reading when no obstacle is within range
    SENSOR_NO_DETECTION_VALUE = 1500.0
    
    # Alignment thresholds
    ALIGNMENT_SAMPLES = 4
    ALIGNMENT_INTERVAL = 50.0
    MAX_ALIGNMENT_ERROR = 30.0


class ParkingConfig:
    """Parallel Parking algorithm maneuver and spatial parameters"""
    # Spatial requirements (cm)
    MIN_PARKING_SPACE = 550.0         # Minimum spot length (Robot length + 150 cm safety margin)
    IDEAL_PARKING_SPACE = 650.0       # Ideal spot length
    MAX_SEARCH_DISTANCE = 3000.0      # Maximum search distance before timing out
    
    # Parallel Parking Kinematic Trajectory Distances (cm)
    DRIVE_PAST_SPOT_DISTANCE = 180.0  # Distance to drive past detected spot to align bumper with front car
    REVERSE_STAGE1_DISTANCE = 220.0   # Step 1: Distance to reverse while turning sharp right into spot (~45 deg)
    REVERSE_STAGE2_DISTANCE = 220.0   # Step 2: Distance to reverse while counter-steering sharp left into slot
    CENTERING_FORWARD_DISTANCE = 60.0  # Step 3: Forward distance to center vehicle in slot
    
    # Tolerances
    STOP_FRONT_TOLERANCE = 100.0      # Target distance from front obstacle when parked
    POSITION_TOLERANCE = 30.0         # Allowed positional variance
    MAX_PARKING_ATTEMPTS = 3
    
    # AprilTag ID (used in ROS2 mode)
    PARKING_TAG_ID = 10


class SpeedConfig:
    """Speed settings for different operational modes"""
    # Base algorithm speeds (-15 to +15 scale)
    SEARCH_SPEED = 8           # Speed during parallel parking space scanning
    ALIGN_SPEED = 5            # Speed during alignment alongside front car
    PARKING_REVERSE_SPEED = 5  # Speed during reversing maneuvers into slot
    CENTERING_SPEED = 3        # Slow speed during final forward/backward adjustment
    CRUISE_SPEED = 12          # Normal driving speed


class TopicConfig:
    """ROS2 Topic names for physical hardware communication"""
    # Subscriptions
    CAMERA_TOPIC = '/camera/image_raw'
    ULTRASONIC_TOPIC = '/ultrasonic/distance'
    ENCODER_TOPIC = '/encoder_pulses'
    
    # Publications
    SERVO_TOPIC = '/servo'
    MOTOR_TOPIC = '/motor_speed'
    
    # Odometry
    ODOMETRY_DISTANCE_TOPIC = '/odometry/distance'
    ODOMETRY_TRIP_TOPIC = '/odometry/trip_distance'
    ODOMETRY_VELOCITY_TOPIC = '/odometry/velocity'
    ODOMETRY_RESET_TOPIC = '/odometry/reset'


class AVISConfig:
    """AVIS Engine Simulator Connection and Sensor Mapping Settings"""
    # TCP Connection
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 25001
    
    # Sensor mapping array: getSensors() returns [left, center, right]
    LEFT_SENSOR_INDEX = 0
    CENTER_SENSOR_INDEX = 1
    RIGHT_SENSOR_INDEX = 2
    
    # Scaling multipliers for AVIS simulator inputs
    STEERING_MULTIPLIER = 4.0   # Multiplier mapping controller steering (-6 to +6) to AVIS (-30 to +30)
    SPEED_MULTIPLIER = 2.0      # Multiplier mapping controller speed (-15 to +15) to AVIS (-30 to +30)
    
    # Odometry conversion factor for simulator loop
    ODOMETRY_SPEED_FACTOR = 1.0
    PARKING_SEARCH_DELAY = 1.0