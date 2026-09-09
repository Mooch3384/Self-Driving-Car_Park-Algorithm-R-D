#!/usr/bin/env python3

"""
Lane Detection System - Improved Curve Following + AprilTag Stop
Version 9: Based on race1.py + AprilTag stopping logic
- Stops on AprilTag ID 6 IF distance < 40cm
- 3 second wait after tag disappears
"""

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

MAX_PIXEL_OFFSET = 150


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
        self.kp = 1.2
        self.kd = 0.25
        self.kc = 1.0  # Increased for more aggressive curve response
        
        # Dynamic ROI expansion when lines are lost
        self.roi_expand_factor = 1.0  # 1.0 = normal, increases when losing lines
        self.MAX_ROI_EXPAND = 1.5  # Maximum expansion factor
        
        self.last_good_steer = 0.0
        self.blind_counter = 0
        self.last_good_left_fit = None
        self.last_good_right_fit = None
        
        self.confidence_left = 0.0
        self.confidence_right = 0.0
        
        self.curvature_radius = float('inf')
        self.curve_direction = 0
        
        self.low_confidence_counter = 0
        self.LOW_CONF_THRESHOLD = 2  # Reduced threshold for faster reaction
        self.MIN_CONFIDENCE = 0.35  # Lower threshold to keep lines longer
        
        self.prev_left_x_bottom = None
        self.prev_right_x_bottom = None
        self.MAX_JUMP_PIXELS = 100  # Allow larger jumps for aggressive curves
        self.MIN_POINTS = 80  # Reduced to keep lines with fewer points
        
        self.force_sliding_window_counter = 0
        self.FORCE_SLIDING_WINDOW_INTERVAL = 10  # More frequent sliding window checks
        
        self.was_in_heavy_turn = False
        self.last_turn_direction = 0
        self.recovery_mode = False
        self.recovery_counter = 0
        self.MAX_RECOVERY_FRAMES = 45  # Increased to hold through longer blind spots

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
        self.roi_expand_factor = 1.0  # Reset ROI expansion

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
        """
        IMPROVED sliding window that follows curves by tracking the tangent direction.
        When pixels are not found, it predicts the next position based on the 
        accumulated slope from previous windows.
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0]//3:, :], axis=0)
        histogram = cv2.GaussianBlur(histogram.astype(np.float32), (51, 1), 0).flatten()
        
        midpoint = int(histogram.shape[0] / 2)
        center_margin = int(self.image_width * 0.08)
        histogram[midpoint - center_margin:midpoint + center_margin] = 0
        
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 18  # More windows for better curve tracking
        base_margin = 100
        margin = int(base_margin * self.roi_expand_factor)  # Dynamic margin expansion
        minpix = 20  # Lower threshold to detect lines in poor conditions
        
        window_height = int(binary_warped.shape[0] // nwindows)
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        # Track x positions and their corresponding y for slope calculation
        left_x_positions = [leftx_current]
        left_y_positions = [binary_warped.shape[0]]
        right_x_positions = [rightx_current]
        right_y_positions = [binary_warped.shape[0]]
        
        # Track consecutive empty windows - stop sooner if we're way off track
        left_empty_count = 0
        right_empty_count = 0
        MAX_EMPTY_BEFORE_STOP = 4  # Stop searching this lane if too many empty windows
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_y_center = (win_y_low + win_y_high) // 2
            
            # Adaptive margin - gets wider when losing track, scales with ROI expansion
            max_margin = int(180 * self.roi_expand_factor)  # Bigger max when expanding ROI
            left_margin = min(margin + left_empty_count * 20, max_margin)
            right_margin = min(margin + right_empty_count * 20, max_margin)
            
            # Window boundaries
            win_xleft_low = max(0, leftx_current - left_margin)
            win_xleft_high = min(binary_warped.shape[1], leftx_current + left_margin)
            win_xright_low = max(0, rightx_current - right_margin)
            win_xright_high = min(binary_warped.shape[1], rightx_current + right_margin)
            
            # Find pixels in windows (only if we haven't given up on this lane)
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
            
            # Update left lane position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
                left_x_positions.append(leftx_current)
                left_y_positions.append(win_y_center)
                left_empty_count = 0
            else:
                left_empty_count += 1
                # PREDICT next position based on curve direction from previous points
                if len(left_x_positions) >= 2:
                    # Calculate slope from last few points
                    dx = left_x_positions[-1] - left_x_positions[-2] if len(left_x_positions) >= 2 else 0
                    dy = left_y_positions[-1] - left_y_positions[-2] if len(left_y_positions) >= 2 else window_height
                    
                    if dy != 0:
                        # Predict x position for next window (moving up means y decreases)
                        slope = dx / dy  # pixels per row
                        predicted_dx = int(slope * window_height)
                        leftx_current = leftx_current + predicted_dx
                        leftx_current = max(0, min(binary_warped.shape[1] - 1, leftx_current))
            
            # Update right lane position
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                right_x_positions.append(rightx_current)
                right_y_positions.append(win_y_center)
                right_empty_count = 0
            else:
                right_empty_count += 1
                # PREDICT next position based on curve direction
                if len(right_x_positions) >= 2:
                    dx = right_x_positions[-1] - right_x_positions[-2] if len(right_x_positions) >= 2 else 0
                    dy = right_y_positions[-1] - right_y_positions[-2] if len(right_y_positions) >= 2 else window_height
                    
                    if dy != 0:
                        slope = dx / dy
                        predicted_dx = int(slope * window_height)
                        rightx_current = rightx_current + predicted_dx
                        rightx_current = max(0, min(binary_warped.shape[1] - 1, rightx_current))
        
        # Concatenate indices - ensure integer type for numpy indexing
        left_lane_inds = np.concatenate(left_lane_inds).astype(np.intp) if left_lane_inds else np.array([], dtype=np.intp)
        right_lane_inds = np.concatenate(right_lane_inds).astype(np.intp) if right_lane_inds else np.array([], dtype=np.intp)
        
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        
        return leftx, lefty, rightx, righty

    def find_lanes_from_prior(self, binary_warped):
        """Search around previous polynomial - better for curve following"""
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
        except Exception:
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
            left_x_top = np.polyval(left_fit, 0)
            
            if left_x_bottom < 0 or left_x_bottom > self.image_width * 0.65:
                left_valid = False
            # Allow more curvature for heavy turns
            if abs(left_fit[0]) > 0.005:
                left_valid = False
            if left_x_top < left_x_bottom - self.image_width * 0.4:
                left_valid = False
                
        if right_fit is not None:
            right_x_bottom = np.polyval(right_fit, y_eval)
            right_x_top = np.polyval(right_fit, 0)
            
            if right_x_bottom > self.image_width or right_x_bottom < self.image_width * 0.35:
                right_valid = False
            if abs(right_fit[0]) > 0.005:
                right_valid = False
            if right_x_top > right_x_bottom + self.image_width * 0.4:
                right_valid = False
        
        if left_fit is not None and right_fit is not None and left_valid and right_valid:
            width_bottom = np.polyval(right_fit, y_eval) - np.polyval(left_fit, y_eval)
            width_top = np.polyval(right_fit, 0) - np.polyval(left_fit, 0)
            
            if width_bottom < self.lane_width * 0.4 or width_bottom > self.lane_width * 1.6:
                if self.confidence_left > self.confidence_right:
                    right_valid = False
                else:
                    left_valid = False
            
            if abs(width_top - width_bottom) > self.lane_width * 0.6:
                if self.confidence_left > self.confidence_right:
                    right_valid = False
                else:
                    left_valid = False
        
        return left_valid, right_valid

    def find_lanes(self, binary_warped, use_prior=True):
        self.force_sliding_window_counter += 1
        
        force_sliding = (
            self.force_sliding_window_counter >= self.FORCE_SLIDING_WINDOW_INTERVAL or
            self.low_confidence_counter >= self.LOW_CONF_THRESHOLD or
            not self.detected or
            self.recovery_mode
        )
        
        if force_sliding:
            self.force_sliding_window_counter = 0
        
        # Prefer polynomial search when we have a good fit (better for curves)
        if use_prior and not force_sliding and self.left_fit is not None and self.right_fit is not None:
            leftx, lefty, rightx, righty = self.find_lanes_from_prior(binary_warped)
            # Fall back to sliding window if not enough points
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
            # Expand ROI gradually when losing lines
            self.roi_expand_factor = min(self.MAX_ROI_EXPAND, self.roi_expand_factor + 0.1)
            if self.low_confidence_counter >= self.LOW_CONF_THRESHOLD:
                self.reset_tracker("Low confidence")
        else:
            self.low_confidence_counter = 0
            # Shrink ROI back to normal when lines are found
            self.roi_expand_factor = max(1.0, self.roi_expand_factor - 0.05)
        
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
            self.right_fit = np.array([
                self.left_fit[0], self.left_fit[1], self.left_fit[2] + self.lane_width
            ])
        elif self.right_fit is not None and self.left_fit is None:
            self.left_fit = np.array([
                self.right_fit[0], self.right_fit[1], self.right_fit[2] - self.lane_width
            ])
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
        
        # MORE AGGRESSIVE curve weights - look further ahead in curves
        if self.curvature_radius < 400:
            weights = [0.15, 0.35, 0.5]  # Very heavy on lookahead for tight curves
        elif self.curvature_radius < 700:
            weights = [0.25, 0.40, 0.35]  # Strong lookahead
        elif self.curvature_radius < 1200:
            weights = [0.35, 0.40, 0.25]
        else:
            weights = [0.55, 0.30, 0.15]
        
        error = (weights[0] * (center_near - car_center) + 
                weights[1] * (center_mid - car_center) + 
                weights[2] * (center_far - car_center))
        
        p_term = error * self.kp
        d_term = (error - self.prev_error) * self.kd
        
        # MORE AGGRESSIVE curvature steering - start rotating earlier
        curvature_steer = 0.0
        if self.curvature_radius < 3000:  # Increased range for earlier response
            # Non-linear response: stronger steering for tighter curves
            curve_factor = (1500 / max(self.curvature_radius, 80)) ** 1.2
            curvature_steer = self.curve_direction * self.kc * curve_factor
        
        final_steer = p_term + d_term + curvature_steer
        steering = np.clip(final_steer / MAX_PIXEL_OFFSET, -1.0, 1.0)
        
        self.prev_error = error
        self.last_good_steer = steering
        return steering

    def is_in_recovery(self):
        return self.recovery_mode or not self.detected


class CudaLaneDetector(Node):
    def __init__(self):
        super().__init__('cuda_lane_detector')
        
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
        
        self.tl = (242, 305)
        self.tr = (665, 305)
        self.br_pt = (944, 540)
        self.bl = (43, 540)
        
        self.calculate_perspective_matrices()
        
        # --- APRILTAG SETUP ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.TAG_SIZE_M = 0.13        # Tag size in Meters (13 cm)
        self.FOCAL_LENGTH = 400.0     # Fixed Focal Length
        self.STOP_DISTANCE_CM = 40.0  # Stop when closer than this
        
        # Define 3D points of the tag (centered at 0,0,0)
        half_size = self.TAG_SIZE_M / 2
        self.tag_obj_points = np.array([
            [-half_size, half_size, 0],
            [ half_size, half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        # --- STOP LOGIC STATE ---
        self.stop_active = False       # True if we have encountered ID 6 and are in "stopped" state
        self.stop_timer_start = None   # Timestamp when the tag disappeared to start countdown
        self.STOP_WAIT_TIME = 3.0      # Seconds to wait after tag disappears
        
        try:
            self.gpu_frame = cv2.cuda_GpuMat()
            self.tracker = LaneTracker(self.max_width, self.max_height)
            self.use_gpu = True
            self.get_logger().info(f'CUDA Lane Detector V9 initialized. Warped: {self.max_width}x{self.max_height}')
        except Exception:
            self.use_gpu = False
            self.tracker = LaneTracker(self.max_width, self.max_height)
            self.get_logger().info('CPU Lane Detection V9 initialized')

        self.last_servo_angle = None
        self.last_speed = None
        
        self.straight_speed = 12
        self.turn_speed = 6
        self.recovery_speed = 4 
        
        self.thresh_val = 205  # Slightly lower threshold for better low-light detection
        self.tracker.kp = 1.2
        self.tracker.kd = 0.25
        self.tracker.kc = 0.9  # More aggressive curve response
        
        self.get_logger().info(f'Threshold: {self.thresh_val}, Kp: 1.2, Kd: 0.25, Kc: 0.9')
        self.get_logger().info('V9: AGGRESSIVE curves + AprilTag Stop (ID 6 < 40cm)')

    def calculate_perspective_matrices(self):
        pts_input = np.float32([self.tl, self.tr, self.bl, self.br_pt])
        
        width_a = np.sqrt(((self.br_pt[0] - self.bl[0]) ** 2) + ((self.br_pt[1] - self.bl[1]) ** 2))
        width_b = np.sqrt(((self.tr[0] - self.tl[0]) ** 2) + ((self.tr[1] - self.tl[1]) ** 2))
        self.max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((self.tr[0] - self.br_pt[0]) ** 2) + ((self.tr[1] - self.br_pt[1]) ** 2))
        height_b = np.sqrt(((self.tl[0] - self.bl[0]) ** 2) + ((self.tl[1] - self.bl[1]) ** 2))
        self.max_height = max(int(height_a), int(height_b))
        
        dst_pts = np.float32([
            [0, 0], 
            [self.max_width - 1, 0], 
            [0, self.max_height - 1], 
            [self.max_width - 1, self.max_height - 1]
        ])
        
        self.M = cv2.getPerspectiveTransform(pts_input, dst_pts)

    def combined_gradient_threshold(self, img):
        """Gradient ONLY Detection (User Request).
        
        Uses Sobel gradient to detect lane edges.
        Binary brightness threshold is REMOVED (or effectively ignored) 
        as per testing showing 'gradient=1, threshold=0' is best.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # === GRADIENT MASK ===
        # Sobel gradient in x direction (detects vertical lane edges)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_val = np.max(abs_sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / (max_val + 1e-6))
        
        # Keep the gradient threshold at 30 as per original
        gradient_mask = (scaled_sobel >= 30).astype(np.uint8) * 255
        
        # === BINARY MASK REMOVED ===
        # Previously: combined_mask = cv2.bitwise_and(gradient_mask, threshold_mask)
        # Now: Just use gradient_mask directly
        
        combined_mask = gradient_mask
        
        # Morphological cleanup to connect lane segments
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask

    def send_servo_command(self, steer_val):
        servo_angle = int(steer_val * -6)
        servo_angle = max(-6, min(6, servo_angle))
        
        if servo_angle != self.last_servo_angle:
            msg_servo = Int8()
            msg_servo.data = servo_angle
            self.servo_pub.publish(msg_servo)
            self.last_servo_angle = servo_angle
        
        return servo_angle

    def send_speed_command(self, servo_angle, is_recovering, should_stop=False):
        if should_stop:
            speed = 0
            # Force log less frequently or just when state changes? 
            # For now, let's just log if it changes, which is handled below.
        elif is_recovering:
            speed = self.recovery_speed
        elif -1 <= servo_angle <= 1:
            speed = self.straight_speed
        else:
            speed = self.turn_speed
        
        if speed != self.last_speed:
            msg_speed = Int8()
            msg_speed.data = speed
            self.speed_pub.publish(msg_speed)
            self.last_speed = speed
            status_msg = f'Speed: {speed}'
            if should_stop:
                status_msg += ' (STOPPED - TAG 6)'
            elif is_recovering:
                status_msg += ' (RECOVERY)'
            self.get_logger().info(status_msg)
        
        return speed

    def listener_callback(self, msg):
        try:
            current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            height, width = current_frame.shape[:2]
            
            # --- APRILTAG DETECTION ---
            corners, ids, rejected = self.detector.detectMarkers(gray)
            
            tag_6_detected = False
            dist_cm = float('inf')
            
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(current_frame, corners, ids)
                
                # Camera Matrix (Fixed)
                cx = width / 2
                cy = height / 2
                camera_matrix = np.array([
                    [self.FOCAL_LENGTH, 0, cx],
                    [0, self.FOCAL_LENGTH, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                dist_coeffs = np.zeros((4,1))
                
                for i, id in enumerate(ids):
                    if id[0] == 6:
                        # Calculate distance
                        img_points = corners[i].reshape(4, 2)
                        success, rvec, tvec = cv2.solvePnP(
                            self.tag_obj_points, 
                            img_points, 
                            camera_matrix, 
                            dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        
                        if success:
                            dist_meters = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
                            dist_cm = dist_meters[0] * 100
                            
                            # Draw distance text
                            center_x = int(np.mean(img_points[:, 0]))
                            center_y = int(np.mean(img_points[:, 1]))
                            cv2.putText(current_frame, f"{dist_cm:.1f}cm", (center_x, center_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            if dist_cm < self.STOP_DISTANCE_CM:
                                tag_6_detected = True
                        break

            # --- STOP LOGIC ---
            should_stop = False
            
            if tag_6_detected:
                self.stop_active = True
                self.stop_timer_start = None # Reset timer if we see the tag
                should_stop = True
                self.get_logger().info(f"Tag 6 Detected at {dist_cm:.1f} cm: STOPPING")
                
            elif self.stop_active:
                # Tag seen previously, but not now. Check timer.
                if self.stop_timer_start is None:
                    self.stop_timer_start = time.time()
                    self.get_logger().info("Tag 6 lost: Starting 3s timer")
                
                elapsed = time.time() - self.stop_timer_start
                if elapsed < self.STOP_WAIT_TIME:
                    should_stop = True
                else:
                    self.stop_active = False # Resume
                    should_stop = False
                    self.get_logger().info("3s Wait Complete: RESUMING")
            
            # --- LANE DETECTION ---
            if self.use_gpu:
                self.gpu_frame.upload(current_frame)
                gpu_warped = cv2.cuda.warpPerspective(
                    self.gpu_frame, self.M, (self.max_width, self.max_height))
                warped = gpu_warped.download()
                binary_mask = self.combined_gradient_threshold(warped)
            else:
                warped = cv2.warpPerspective(current_frame, self.M, (self.max_width, self.max_height))
                binary_mask = self.combined_gradient_threshold(warped)
            
            self.tracker.find_lanes(binary_mask)
            steer_val = self.tracker.calculate_steering()

            is_recovering = self.tracker.is_in_recovery()
            
            # If stopping, we still might want to steer? 
            # Usually better to hold steering or keep centering if possible, 
            # but if we are stopped, steering doesn't do much. 
            # We will continue to calculate steering so ready when resizing.
            
            servo_angle = self.send_servo_command(steer_val)
            self.send_speed_command(servo_angle, is_recovering, should_stop=should_stop)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = CudaLaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        msg_speed = Int8()
        msg_speed.data = 0
        node.speed_pub.publish(msg_speed)
        
        msg_servo = Int8()
        msg_servo.data = 0
        node.servo_pub.publish(msg_servo)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
