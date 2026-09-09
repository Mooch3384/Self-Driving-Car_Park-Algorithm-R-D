#!/usr/bin/env python3

"""
Lane Detection + Traffic Sign Detection System
Version 10: Adds distance measurement to detected signs
- When stop sign detected: speed = 0
- After 5 seconds without stop sign: resume normal operation
- Distance estimation based on sign bounding box size
"""

# ============================================================
# DISPLAY TOGGLE: Set to True to show camera, False for headless
# ============================================================
SHOW_DISPLAY = False
# ============================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import time
import os

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

MAX_PIXEL_OFFSET = 150


class TRTInference:
    """TensorRT inference wrapper"""
    def __init__(self, engine_path):
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


class LaneTracker:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        
        self.lane_width = int(image_width * 0.75)
        self.detected = False
        self.left_detected = False
        self.right_detected = False
        
        self.prev_error = 0.0
        self.kp = 1.0
        self.kd = 0.2
        self.kc = 0.5
        
        self.last_good_steer = 0.0
        self.blind_counter = 0
        self.last_good_left_fit = None
        self.last_good_right_fit = None
        
        self.confidence_left = 0.0
        self.confidence_right = 0.0
        
        self.curvature_radius = float('inf')
        self.curve_direction = 0
        
        self.low_confidence_counter = 0
        self.LOW_CONF_THRESHOLD = 3
        self.MIN_CONFIDENCE = 0.1
        
        self.prev_left_x_bottom = None
        self.prev_right_x_bottom = None
        self.MAX_JUMP_PIXELS = 80
        self.MIN_POINTS = 100
        
        self.force_sliding_window_counter = 0
        self.FORCE_SLIDING_WINDOW_INTERVAL = 15
        
        self.was_in_heavy_turn = False
        self.last_turn_direction = 0
        self.recovery_mode = False
        self.recovery_counter = 0
        self.MAX_RECOVERY_FRAMES = 30

    def reset_tracker(self, reason=""):
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history.clear()
        self.right_fit_history.clear()
        self.detected = False
        self.left_detected = False
        self.right_detected = False
        self.confidence_left = 0.0
        self.confidence_right = 0.0
        self.low_confidence_counter = 0
        self.prev_left_x_bottom = None
        self.prev_right_x_bottom = None
        self.recovery_mode = False

    def detect_lane_jump(self, new_left_fit, new_right_fit):
        y_eval = self.image_height
        jump_detected = False
        if new_left_fit is not None and self.prev_left_x_bottom is not None:
            new_left_x = np.polyval(new_left_fit, y_eval)
            if abs(new_left_x - self.prev_left_x_bottom) > self.MAX_JUMP_PIXELS:
                jump_detected = True
        if new_right_fit is not None and self.prev_right_x_bottom is not None:
            new_right_x = np.polyval(new_right_fit, y_eval)
            if abs(new_right_x - self.prev_right_x_bottom) > self.MAX_JUMP_PIXELS:
                jump_detected = True
        return jump_detected

    def update_position_tracking(self):
        y_eval = self.image_height
        if self.left_fit is not None:
            self.prev_left_x_bottom = np.polyval(self.left_fit, y_eval)
        if self.right_fit is not None:
            self.prev_right_x_bottom = np.polyval(self.right_fit, y_eval)

    def find_lanes_sliding_window(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//3:, :], axis=0)
        histogram = cv2.GaussianBlur(histogram.astype(np.float32), (51, 1), 0).flatten()
        
        midpoint = int(histogram.shape[0] / 2)
        center_margin = int(self.image_width * 0.08)
        histogram[midpoint - center_margin:midpoint + center_margin] = 0
        
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 12
        margin = 100
        minpix = 25
        window_height = int(binary_warped.shape[0] // nwindows)
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        left_x_positions = [leftx_current]
        left_y_positions = [binary_warped.shape[0]]
        right_x_positions = [rightx_current]
        right_y_positions = [binary_warped.shape[0]]
        
        left_empty_count = 0
        right_empty_count = 0
        MAX_EMPTY_BEFORE_STOP = 4
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_y_center = (win_y_low + win_y_high) // 2
            
            left_margin = min(margin + left_empty_count * 15, 150)
            right_margin = min(margin + right_empty_count * 15, 150)
            
            win_xleft_low = max(0, leftx_current - left_margin)
            win_xleft_high = min(binary_warped.shape[1], leftx_current + left_margin)
            win_xright_low = max(0, rightx_current - right_margin)
            win_xright_high = min(binary_warped.shape[1], rightx_current + right_margin)
            
            if left_empty_count < MAX_EMPTY_BEFORE_STOP:
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                 (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            else:
                good_left_inds = np.array([])
                
            if right_empty_count < MAX_EMPTY_BEFORE_STOP:
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                  (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            else:
                good_right_inds = np.array([])
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
                left_x_positions.append(leftx_current)
                left_y_positions.append(win_y_center)
                left_empty_count = 0
            else:
                left_empty_count += 1
                if len(left_x_positions) >= 2:
                    dx = left_x_positions[-1] - left_x_positions[-2]
                    dy = left_y_positions[-1] - left_y_positions[-2] if len(left_y_positions) >= 2 else window_height
                    if dy != 0:
                        slope = dx / dy
                        predicted_dx = int(slope * window_height)
                        leftx_current = max(0, min(binary_warped.shape[1] - 1, leftx_current + predicted_dx))
            
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                right_x_positions.append(rightx_current)
                right_y_positions.append(win_y_center)
                right_empty_count = 0
            else:
                right_empty_count += 1
                if len(right_x_positions) >= 2:
                    dx = right_x_positions[-1] - right_x_positions[-2]
                    dy = right_y_positions[-1] - right_y_positions[-2] if len(right_y_positions) >= 2 else window_height
                    if dy != 0:
                        slope = dx / dy
                        predicted_dx = int(slope * window_height)
                        rightx_current = max(0, min(binary_warped.shape[1] - 1, rightx_current + predicted_dx))
        
        left_lane_inds = np.concatenate(left_lane_inds).astype(np.intp) if left_lane_inds else np.array([], dtype=np.intp)
        right_lane_inds = np.concatenate(right_lane_inds).astype(np.intp) if right_lane_inds else np.array([], dtype=np.intp)
        
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        
        return leftx, lefty, rightx, righty

    def find_lanes_from_prior(self, binary_warped):
        margin = 80
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx, lefty = np.array([]), np.array([])
        rightx, righty = np.array([]), np.array([])
        
        if self.left_fit is not None:
            left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + 
                              self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & 
                             (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                              self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
        
        if self.right_fit is not None:
            right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + 
                               self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & 
                              (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                               self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty

    def fit_polynomial(self, x, y):
        if len(x) < self.MIN_POINTS:
            return None, 0.0
        try:
            weights = y / self.image_height
            fit = np.polyfit(y, x, 2, w=weights)
            fitted_x = np.polyval(fit, y)
            residuals = np.abs(x - fitted_x)
            mean_residual = np.mean(residuals)
            confidence = max(0, 1.0 - mean_residual / 50)
            return fit, confidence
        except:
            return None, 0.0

    def smooth_fit(self, new_fit, history, confidence):
        if new_fit is None:
            if len(history) > 0:
                return np.mean(history, axis=0)
            return None
        if confidence > self.MIN_CONFIDENCE:
            history.append(new_fit)
        else:
            if len(history) > 0:
                return np.mean(history, axis=0)
            return new_fit
        if len(history) == 0:
            return new_fit
        weights = np.linspace(0.5, 1.0, len(history))
        weights /= weights.sum()
        return np.average(history, axis=0, weights=weights)

    def validate_lanes(self, left_fit, right_fit):
        if left_fit is None and right_fit is None:
            return False, False
        y_eval = self.image_height
        left_valid = True
        right_valid = True
        
        if left_fit is not None:
            left_x_bottom = np.polyval(left_fit, y_eval)
            if left_x_bottom < 0 or left_x_bottom > self.image_width * 0.65:
                left_valid = False
            if abs(left_fit[0]) > 0.005:
                left_valid = False
                
        if right_fit is not None:
            right_x_bottom = np.polyval(right_fit, y_eval)
            if right_x_bottom > self.image_width or right_x_bottom < self.image_width * 0.35:
                right_valid = False
            if abs(right_fit[0]) > 0.005:
                right_valid = False
        
        return left_valid, right_valid

    def find_lanes(self, binary_warped, use_prior=True):
        self.force_sliding_window_counter += 1
        force_sliding = (
            self.force_sliding_window_counter >= self.FORCE_SLIDING_WINDOW_INTERVAL or
            self.low_confidence_counter >= self.LOW_CONF_THRESHOLD or
            not self.detected or self.recovery_mode
        )
        if force_sliding:
            self.force_sliding_window_counter = 0
        
        if use_prior and not force_sliding and self.left_fit is not None and self.right_fit is not None:
            leftx, lefty, rightx, righty = self.find_lanes_from_prior(binary_warped)
            if len(leftx) < 300 or len(rightx) < 300:
                leftx, lefty, rightx, righty = self.find_lanes_sliding_window(binary_warped)
        else:
            leftx, lefty, rightx, righty = self.find_lanes_sliding_window(binary_warped)
        
        left_fit_new, conf_left = self.fit_polynomial(leftx, lefty)
        right_fit_new, conf_right = self.fit_polynomial(rightx, righty)
        
        if self.detect_lane_jump(left_fit_new, right_fit_new):
            conf_left *= 0.3
            conf_right *= 0.3
        
        self.confidence_left = conf_left
        self.confidence_right = conf_right
        
        if conf_left < self.MIN_CONFIDENCE and conf_right < self.MIN_CONFIDENCE:
            self.low_confidence_counter += 1
            if self.low_confidence_counter >= self.LOW_CONF_THRESHOLD:
                self.reset_tracker("Low confidence")
        else:
            self.low_confidence_counter = 0
        
        left_valid, right_valid = self.validate_lanes(left_fit_new, right_fit_new)
        if not left_valid:
            left_fit_new = None
            conf_left = 0.0
        if not right_valid:
            right_fit_new = None
            conf_right = 0.0
        
        self.left_fit = self.smooth_fit(left_fit_new, self.left_fit_history, conf_left)
        self.right_fit = self.smooth_fit(right_fit_new, self.right_fit_history, conf_right)
        self._handle_missing_lanes()
        
        self.left_detected = self.left_fit is not None
        self.right_detected = self.right_fit is not None
        self.detected = self.left_detected or self.right_detected
        
        if self.left_detected and conf_left > 0.6:
            self.last_good_left_fit = self.left_fit.copy()
        if self.right_detected and conf_right > 0.6:
            self.last_good_right_fit = self.right_fit.copy()
        
        self.update_position_tracking()
        if self.detected:
            self._calculate_curvature()
            if self.recovery_mode:
                self.recovery_mode = False
                self.recovery_counter = 0

    def _handle_missing_lanes(self):
        if self.left_fit is not None and self.right_fit is None:
            self.right_fit = np.array([self.left_fit[0], self.left_fit[1], self.left_fit[2] + self.lane_width])
        elif self.right_fit is not None and self.left_fit is None:
            self.left_fit = np.array([self.right_fit[0], self.right_fit[1], self.right_fit[2] - self.lane_width])
        elif self.left_fit is None and self.right_fit is None:
            if self.last_good_left_fit is not None and self.blind_counter < 10:
                self.left_fit = self.last_good_left_fit
            if self.last_good_right_fit is not None and self.blind_counter < 10:
                self.right_fit = self.last_good_right_fit

    def _calculate_curvature(self):
        if self.left_fit is None and self.right_fit is None:
            self.curvature_radius = float('inf')
            self.curve_direction = 0
            return
        y_eval = self.image_height
        fit = self.left_fit if self.left_fit is not None else self.right_fit
        A, B = fit[0], fit[1]
        if abs(A) < 1e-6:
            self.curvature_radius = float('inf')
            self.curve_direction = 0
        else:
            numerator = (1 + (2*A*y_eval + B)**2)**1.5
            self.curvature_radius = abs(numerator / (2*A))
            if A < -0.0001:
                self.curve_direction = 1
            elif A > 0.0001:
                self.curve_direction = -1
            else:
                self.curve_direction = 0

    def calculate_steering(self):
        if not self.detected:
            self.blind_counter += 1
            if not self.recovery_mode:
                self.recovery_mode = True
                self.recovery_counter = 0
                if abs(self.last_good_steer) > 0.2:
                    self.last_turn_direction = 1 if self.last_good_steer > 0 else -1
                    self.was_in_heavy_turn = True
                else:
                    self.was_in_heavy_turn = False
            self.recovery_counter += 1
            if self.recovery_counter < self.MAX_RECOVERY_FRAMES:
                if self.was_in_heavy_turn:
                    return self.last_turn_direction * 0.6
                else:
                    return self.last_good_steer * max(0.8, 1.0 - self.blind_counter * 0.03)
            else:
                self.recovery_mode = False
                return 0.0
        
        self.blind_counter = 0
        if self.curvature_radius < 500:
            self.was_in_heavy_turn = True
            self.last_turn_direction = self.curve_direction
        
        y_near = self.image_height
        y_mid = int(self.image_height * 0.7)
        y_far = int(self.image_height * 0.4)
        
        def lane_center_at(y):
            left_x = np.polyval(self.left_fit, y) if self.left_fit is not None else 0
            right_x = np.polyval(self.right_fit, y) if self.right_fit is not None else self.image_width
            return (left_x + right_x) / 2
        
        center_near = lane_center_at(y_near)
        center_mid = lane_center_at(y_mid)
        center_far = lane_center_at(y_far)
        car_center = self.image_width / 2
        
        if self.curvature_radius < 500:
            weights = [0.3, 0.4, 0.3]
        elif self.curvature_radius < 1000:
            weights = [0.4, 0.4, 0.2]
        else:
            weights = [0.6, 0.3, 0.1]
        
        error = (weights[0] * (center_near - car_center) + 
                weights[1] * (center_mid - car_center) + 
                weights[2] * (center_far - car_center))
        
        p_term = error * self.kp
        d_term = (error - self.prev_error) * self.kd
        curvature_steer = 0.0
        if self.curvature_radius < 2000:
            curvature_steer = self.curve_direction * self.kc * (1000 / max(self.curvature_radius, 100))
        
        final_steer = p_term + d_term + curvature_steer
        steering = np.clip(final_steer / MAX_PIXEL_OFFSET, -1.0, 1.0)
        self.prev_error = error
        self.last_good_steer = steering
        return steering

    def is_in_recovery(self):
        return self.recovery_mode or not self.detected

    def get_lane_points(self):
        """Get lane polynomial points for visualization"""
        if self.left_fit is None and self.right_fit is None:
            return None, None
        ploty = np.linspace(0, self.image_height - 1, self.image_height)
        left_fitx = None
        right_fitx = None
        if self.left_fit is not None:
            left_fitx = np.polyval(self.left_fit, ploty)
            left_fitx = np.clip(left_fitx, 0, self.image_width - 1)
        if self.right_fit is not None:
            right_fitx = np.polyval(self.right_fit, ploty)
            right_fitx = np.clip(right_fitx, 0, self.image_width - 1)
        return (left_fitx, ploty), (right_fitx, ploty)


class LaneAndSignDetector(Node):
    def __init__(self):
        super().__init__('lane_and_sign_detector')
        
        realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.listener_callback, 10)
        
        self.servo_pub = self.create_publisher(Int8, '/servo', realtime_qos)
        self.speed_pub = self.create_publisher(Int8, '/motor_speed', realtime_qos)
        
        self.br = CvBridge()
        
        # Lane detection setup
        self.tl = (242, 305)
        self.tr = (665, 305)
        self.br_pt = (944, 540)
        self.bl = (43, 540)
        self.calculate_perspective_matrices()
        
        try:
            self.gpu_frame = cv2.cuda_GpuMat()
            self.tracker = LaneTracker(self.max_width, self.max_height)
            self.use_gpu = True
        except:
            self.use_gpu = False
            self.tracker = LaneTracker(self.max_width, self.max_height)
        
        # Sign detection setup
        self.engine_path = "/home/jetson/Desktop/anti/best.engine"
        self.template_dir = "/home/jetson/Desktop/anti/templates"
        self.conf_threshold = 0.3
        self.input_size = (640, 640)
        self.templates = {}
        self.template_grays = {}
        self.template_size = (100, 100)
        
        try:
            self.trt_engine = TRTInference(self.engine_path)
            self.get_logger().info("TensorRT engine loaded!")
        except Exception as e:
            self.get_logger().error(f"TRT load failed: {e}")
            self.trt_engine = None
        
        self.load_templates()
        
        # Speed settings
        self.last_servo_angle = None
        self.last_speed = None
        self.straight_speed = 12
        self.turn_speed = 6
        self.recovery_speed = 4
        self.thresh_val = 150
        
        # Stop sign handling
        self.stop_sign_detected = False
        self.last_stop_time = 0
        self.STOP_TIMEOUT = 5.0  # Resume after 5 seconds without stop sign
        
        # Sign colors
        self.sign_colors = {
            'forward': (0, 255, 0),
            'left': (255, 0, 0),
            'right': (0, 0, 255),
            'stop': (0, 255, 255),
            'unknown': (128, 128, 128)
        }
        
        # FPS tracking
        self.last_fps_time = cv2.getTickCount()
        self.fps = 0
        
        # Display window
        if SHOW_DISPLAY:
            cv2.namedWindow('Lane + Sign Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Lane + Sign Detection', 960, 540)
        
        # Distance estimation parameters (calibrate these for your setup)
        # REAL_SIGN_HEIGHT_CM: Actual height of your traffic signs in centimeters
        # FOCAL_LENGTH_PIXELS: Camera focal length in pixels (calibrate by measuring)
        # Formula: distance = (real_height * focal_length) / pixel_height
        self.REAL_SIGN_HEIGHT_CM = 15.0  # Real sign height in cm (adjust to your sign)
        self.FOCAL_LENGTH_PIXELS = 400.0  # Calibrated based on 30cm actual = 60cm shown (halved)
        # To calibrate: place sign at known distance, measure pixel height
        # focal_length = (pixel_height * known_distance) / real_height
        
        self.get_logger().info('V10: Lane + Sign Detection with Distance')
        self.get_logger().info('Stop sign = speed 0, resume after 5s without stop')

    def load_templates(self):
        for sign_type in ['forward', 'left', 'right', 'stop']:
            path = os.path.join(self.template_dir, f"{sign_type}.png")
            if os.path.exists(path):
                template = cv2.imread(path)
                if template is not None:
                    template = cv2.resize(template, self.template_size)
                    self.templates[sign_type] = template
                    self.template_grays[sign_type] = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    self.get_logger().info(f"Loaded template: {sign_type}")

    def calculate_perspective_matrices(self):
        pts_input = np.float32([self.tl, self.tr, self.bl, self.br_pt])
        width_a = np.sqrt(((self.br_pt[0] - self.bl[0]) ** 2) + ((self.br_pt[1] - self.bl[1]) ** 2))
        width_b = np.sqrt(((self.tr[0] - self.tl[0]) ** 2) + ((self.tr[1] - self.tl[1]) ** 2))
        self.max_width = max(int(width_a), int(width_b))
        height_a = np.sqrt(((self.tr[0] - self.br_pt[0]) ** 2) + ((self.tr[1] - self.br_pt[1]) ** 2))
        height_b = np.sqrt(((self.tl[0] - self.bl[0]) ** 2) + ((self.tl[1] - self.bl[1]) ** 2))
        self.max_height = max(int(height_a), int(height_b))
        dst_pts = np.float32([[0, 0], [self.max_width - 1, 0], 
                              [0, self.max_height - 1], [self.max_width - 1, self.max_height - 1]])
        self.M = cv2.getPerspectiveTransform(pts_input, dst_pts)

    def detect_signs(self, frame):
        """Run sign detection and return list of detected signs"""
        if self.trt_engine is None:
            return []
        
        # Preprocess
        self.orig_h, self.orig_w = frame.shape[:2]
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(np.expand_dims(img, axis=0))
        
        # Inference
        outputs = self.trt_engine.infer(img)
        detections = outputs[0]
        
        scale_x = self.orig_w / self.input_size[0]
        scale_y = self.orig_h / self.input_size[1]
        
        signs = []
        for det in detections[0]:
            x1, y1, x2, y2, conf, cls_id = det
            if conf >= self.conf_threshold:
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                region = frame[y1:y2, x1:x2]
                if region.size > 0:
                    sign_type = self.classify_region(region)
                    height_pixels = y2 - y1
                    distance_cm = self.calculate_distance(height_pixels)
                    signs.append({
                        'type': sign_type, 
                        'box': [x1, y1, x2, y2], 
                        'conf': conf,
                        'distance': distance_cm
                    })
        
        return signs

    def calculate_distance(self, pixel_height):
        """Calculate distance to sign based on its pixel height using pinhole camera model"""
        if pixel_height <= 0:
            return float('inf')
        # distance = (real_height * focal_length) / pixel_height
        distance_cm = (self.REAL_SIGN_HEIGHT_CM * self.FOCAL_LENGTH_PIXELS) / pixel_height
        return distance_cm

    def classify_region(self, region):
        """Classify region using template matching"""
        if not self.template_grays:
            return 'unknown'
        
        best_match = 'unknown'
        best_score = 0.0
        
        region_resized = cv2.resize(region, self.template_size)
        region_gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
        
        for sign_type, template_gray in self.template_grays.items():
            try:
                result = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                if score > best_score:
                    best_score = score
                    best_match = sign_type
            except:
                pass
        
        return best_match if best_score > 0.3 else 'unknown'

    def send_servo_command(self, steer_val):
        servo_angle = int(steer_val * -6)
        servo_angle = max(-6, min(6, servo_angle))
        if servo_angle != self.last_servo_angle:
            msg = Int8()
            msg.data = servo_angle
            self.servo_pub.publish(msg)
            self.last_servo_angle = servo_angle
        return servo_angle

    def send_speed_command(self, speed):
        if speed != self.last_speed:
            msg = Int8()
            msg.data = speed
            self.speed_pub.publish(msg)
            self.last_speed = speed
            self.get_logger().info(f'Speed: {speed}')

    def listener_callback(self, msg):
        try:
            current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_time = time.time()
            
            # Sign detection
            signs = self.detect_signs(current_frame)
            
            # Check for stop sign within 30cm
            stop_sign_close = False
            for sign in signs:
                if sign['type'] == 'stop' and sign.get('distance', float('inf')) < 30.0:
                    stop_sign_close = True
                    break
            
            # Update stop state based on distance
            if stop_sign_close:
                # Sign is close - stop and reset timer
                if not self.stop_sign_detected:
                    self.get_logger().info('STOP SIGN < 30cm - Stopping')
                self.stop_sign_detected = True
                self.last_stop_time = current_time  # Reset timer while sign is visible
            elif self.stop_sign_detected:
                # Sign was detected but now removed - wait 5 seconds before resuming
                if current_time - self.last_stop_time > self.STOP_TIMEOUT:
                    self.stop_sign_detected = False
                    self.get_logger().info('5s after sign removed - Resuming')
            
            # Lane detection
            if self.use_gpu:
                self.gpu_frame.upload(current_frame)
                gpu_warped = cv2.cuda.warpPerspective(self.gpu_frame, self.M, (self.max_width, self.max_height))
                gpu_gray = cv2.cuda.cvtColor(gpu_warped, cv2.COLOR_BGR2GRAY)
                _, gpu_mask = cv2.cuda.threshold(gpu_gray, self.thresh_val, 255, cv2.THRESH_BINARY)
                binary_mask = gpu_mask.download()
            else:
                warped = cv2.warpPerspective(current_frame, self.M, (self.max_width, self.max_height))
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray, self.thresh_val, 255, cv2.THRESH_BINARY)
            
            self.tracker.find_lanes(binary_mask)
            steer_val = self.tracker.calculate_steering()
            
            # Send commands
            servo_angle = self.send_servo_command(steer_val)
            
            # Speed control based on stop sign
            if self.stop_sign_detected:
                self.send_speed_command(0)  # Stop
            else:
                # Normal speed logic
                is_recovering = self.tracker.is_in_recovery()
                if is_recovering:
                    speed = self.recovery_speed
                elif -1 <= servo_angle <= 1:
                    speed = self.straight_speed
                else:
                    speed = self.turn_speed
                self.send_speed_command(speed)
            
            # Visualization
            if SHOW_DISPLAY:
                display_frame = self.draw_visualization(current_frame, signs, servo_angle)
                cv2.imshow('Lane + Sign Detection', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def draw_visualization(self, frame, signs, servo_angle):
        """Draw lane lines, sign boxes, and status on frame"""
        display = frame.copy()
        
        # Draw ROI polygon
        pts = np.array([self.tl, self.tr, self.br_pt, self.bl], np.int32)
        cv2.polylines(display, [pts.reshape((-1, 1, 2))], True, (255, 255, 0), 2)
        
        # Draw lane overlay on warped region
        left_pts, right_pts = self.tracker.get_lane_points()
        if left_pts is not None or right_pts is not None:
            # Create overlay for lane visualization
            h, w = frame.shape[:2]
            if left_pts is not None and right_pts is not None:
                left_fitx, ploty = left_pts
                right_fitx, _ = right_pts
                if left_fitx is not None and right_fitx is not None:
                    # Draw lane center line on original frame (approximate)
                    for i in range(0, len(ploty) - 20, 20):
                        # Map warped coords roughly to original
                        y_ratio = ploty[i] / self.max_height
                        center_x = int((left_fitx[i] + right_fitx[i]) / 2)
                        # Draw on lower portion of frame
                        draw_y = int(self.tl[1] + y_ratio * (self.bl[1] - self.tl[1]))
                        x_ratio = center_x / self.max_width
                        draw_x = int(self.tl[0] + x_ratio * (self.tr[0] - self.tl[0]))
                        cv2.circle(display, (draw_x, draw_y), 5, (0, 255, 0), -1)
        
        # Draw sign detections
        for sign in signs:
            x1, y1, x2, y2 = sign['box']
            sign_type = sign['type']
            color = self.sign_colors.get(sign_type, (255, 255, 255))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            
            # Show type, confidence and distance
            distance = sign.get('distance', 0)
            if distance < 100:
                label = f"{sign_type}: {distance:.0f}cm"
            else:
                label = f"{sign_type}: {distance/100:.1f}m"
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_fps_time) / cv2.getTickFrequency()
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.last_fps_time = current_time
        
        # Status overlay
        cv2.putText(display, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Stop sign status
        if self.stop_sign_detected:
            cv2.putText(display, "STOPPED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            remaining = max(0, self.STOP_TIMEOUT - (time.time() - self.last_stop_time))
            cv2.putText(display, f"Resume in: {remaining:.1f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            status = "LANE TRACKING" if self.tracker.detected else "SEARCHING"
            color = (0, 255, 0) if self.tracker.detected else (0, 165, 255)
            cv2.putText(display, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Steering indicator
        cv2.putText(display, f"Steer: {servo_angle}", (10, display.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display


def main(args=None):
    rclpy.init(args=args)
    node = LaneAndSignDetector()
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
        except:
            pass
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
