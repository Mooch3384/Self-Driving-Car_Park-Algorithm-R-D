#!/usr/bin/env python3
"""
Main Integrated ROS2 Node: Autonomous Driving & Parallel Parking Integration
Subscribes to camera, ultrasonic, and odometry topics; controls motor speed and steering.
"""

# Display toggle
SHOW_DISPLAY = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Int32, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

# Project imports
from config import (RobotConfig, SensorConfig, ParkingConfig, 
                   SpeedConfig, TopicConfig)
from parking_controller import ParallelParkingController, ParkingState

# TensorRT (optional inference)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# AprilTag Detector (optional)
try:
    from dt_apriltags import Detector as AprilTagDetector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False


class TRTInference:
    """TensorRT inference wrapper for object / sign detection"""
    
    def __init__(self, engine_path):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT unavailable on this system")
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = 1
            for dim in shape:
                size *= abs(dim)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'].reshape(out['shape']) for out in self.outputs]


class MainNode(Node):
    """Main ROS2 Node integrating vehicle control, perception, and parallel parking"""
    
    def __init__(self):
        super().__init__('main_node')
        
        # QoS Profile
        realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # Subscribers & Publishers
        # ═══════════════════════════════════════════════════════════════════
        
        self.camera_sub = self.create_subscription(
            Image, TopicConfig.CAMERA_TOPIC, self.camera_callback, 10
        )
        self.ultrasonic_sub = self.create_subscription(
            Float32, TopicConfig.ULTRASONIC_TOPIC, self.ultrasonic_callback, realtime_qos
        )
        self.odometry_sub = self.create_subscription(
            Float32, TopicConfig.ODOMETRY_DISTANCE_TOPIC, self.odometry_callback, realtime_qos
        )
        
        self.servo_pub = self.create_publisher(Int8, TopicConfig.SERVO_TOPIC, realtime_qos)
        self.speed_pub = self.create_publisher(Int8, TopicConfig.MOTOR_TOPIC, realtime_qos)
        self.odom_reset_pub = self.create_publisher(Int32, TopicConfig.ODOMETRY_RESET_TOPIC, 10)
        
        self.bridge = CvBridge()
        
        # Sensor Telemetry
        self.ultrasonic_distance = float('inf')
        self.odometry_distance = 0.0
        
        # Operational Mode: 'cruise' or 'parking'
        self.mode = 'cruise'
        
        # Controllers
        self.parking_controller = ParallelParkingController(self.get_logger())
        
        # AprilTag Detector Init
        self.apriltag_detector = None
        if APRILTAG_AVAILABLE:
            try:
                self.apriltag_detector = AprilTagDetector(families='tag36h11', nthreads=2, quad_decimate=2.0)
                self.get_logger().info("AprilTag detector initialized successfully.")
            except Exception as e:
                self.get_logger().warn(f"AprilTag initialization failed: {e}")
        
        # TensorRT Engine Init
        self.declare_parameter('engine_path', 'models/best.engine')
        engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.trt_engine = None
        if TRT_AVAILABLE and os.path.exists(engine_path):
            try:
                self.trt_engine = TRTInference(engine_path)
                self.get_logger().info(f"TensorRT engine loaded from {engine_path}")
            except Exception as e:
                self.get_logger().warn(f"TensorRT model load failed: {e}")
        
        self.last_servo = None
        self.last_speed = None
        
        if SHOW_DISPLAY:
            cv2.namedWindow('Robot View', cv2.WINDOW_NORMAL)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("Main Vehicle ROS2 Node Started")
        self.get_logger().info(f"  Mode: {self.mode}")
        self.get_logger().info(f"  TensorRT Available: {TRT_AVAILABLE}")
        self.get_logger().info(f"  AprilTag Available: {APRILTAG_AVAILABLE}")
        self.get_logger().info("=" * 50)

    # ═══════════════════════════════════════════════════════════════════════
    # TELEMETRY CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════
    
    def ultrasonic_callback(self, msg: Float32):
        self.ultrasonic_distance = msg.data

    def odometry_callback(self, msg: Float32):
        self.odometry_distance = msg.data

    # ═══════════════════════════════════════════════════════════════════════
    # COMMAND PUBLISHERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def send_servo(self, angle: int):
        angle = max(-6, min(6, int(angle)))
        if angle != self.last_servo:
            msg = Int8()
            msg.data = angle
            self.servo_pub.publish(msg)
            self.last_servo = angle

    def send_speed(self, speed: int):
        speed = max(-15, min(15, int(speed)))
        if speed != self.last_speed:
            msg = Int8()
            msg.data = speed
            self.speed_pub.publish(msg)
            self.last_speed = speed

    def reset_odometry(self):
        msg = Int32()
        msg.data = 1
        self.odom_reset_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════════════════
    # PERCEPTION & PROCESSING
    # ═══════════════════════════════════════════════════════════════════════
    
    def detect_parking_tag(self, frame) -> bool:
        if self.apriltag_detector is None:
            return False
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.apriltag_detector.detect(gray)
            for det in detections:
                if det.tag_id == ParkingConfig.PARKING_TAG_ID:
                    return True
        except Exception:
            pass
        return False

    def camera_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Check for parking tag trigger
            if self.detect_parking_tag(frame) and self.mode != 'parking':
                self.get_logger().info("🎯 PARKING TAG DETECTED! Triggering parallel parking mode.")
                self.mode = 'parking'
                self.reset_odometry()
                self.parking_controller.enable()
            
            # Processing by Mode
            if self.mode == 'parking':
                # Sensors array [left, front, right]
                sensors = [float('inf'), self.ultrasonic_distance, float('inf')]
                steering, speed = self.parking_controller.update(sensors, self.odometry_distance)
                
                self.send_servo(steering)
                self.send_speed(speed)
                
                if self.parking_controller.is_completed():
                    self.get_logger().info("Parallel Parking Completed. Returning to cruise mode.")
                    self.mode = 'cruise'
                elif self.parking_controller.is_failed():
                    self.get_logger().warn("Parallel Parking Failed. Returning to cruise mode.")
                    self.mode = 'cruise'
            else:
                # Cruise driving mode
                self.send_servo(0)
                self.send_speed(SpeedConfig.CRUISE_SPEED)
            
            if SHOW_DISPLAY:
                self.draw_display(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    rclpy.shutdown()
                elif key == ord('p'):
                    self.mode = 'parking'
                    self.reset_odometry()
                    self.parking_controller.enable()
                elif key == ord('c'):
                    self.mode = 'cruise'
                    self.parking_controller.disable()
                    
        except Exception as e:
            self.get_logger().error(f"Camera Callback Error: {e}")

    def draw_display(self, frame):
        display = frame.copy()
        cv2.rectangle(display, (0, 0), (320, 140), (0, 0, 0), -1)
        
        y = 25
        cv2.putText(display, f"Mode: {self.mode.upper()}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        cv2.putText(display, f"Ultrasonic: {self.ultrasonic_distance:.1f} cm", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        y += 25
        cv2.putText(display, f"Odometry: {self.odometry_distance:.1f} cm", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        
        if self.mode == 'parking':
            status = self.parking_controller.get_status()
            y += 25
            cv2.putText(display, f"Park State: {status['state']}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        
        cv2.imshow('Robot View', display)


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            msg = Int8()
            msg.data = 0
            node.speed_pub.publish(msg)
            node.servo_pub.publish(msg)
        except Exception:
            pass
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()