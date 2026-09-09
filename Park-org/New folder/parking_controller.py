#!/usr/bin/env python3
"""
Autonomous Vehicle Parking Controllers
Implements both Parallel Parking and Double Parking State Machine Controllers.
"""

from enum import Enum
import time
from config import RobotConfig, SensorConfig, ParkingConfig, SpeedConfig


class ParkingState(Enum):
    """Unified Parking FSM States"""
    DISABLED = 0
    IDLE = 1
    
    # Approach / Drive States
    DRIVE_FORWARD = 10
    SLOW_APPROACH = 11
    SEARCHING_SPOT = 12
    SPOT_DETECTED = 13
    DRIVE_PAST_SPOT = 14
    ALIGN_WITH_FRONT = 15
    
    # Parallel Reversing Maneuver States
    REVERSE_STAGE_1 = 20    # Steer sharp right into spot (~45 deg)
    REVERSE_STAGE_2 = 21    # Counter-steer sharp left into spot to straighten
    
    # Adjustment & Centering States
    STOPPING = 25
    CHECK_REAR_DISTANCE = 26
    ADJUST_BACKWARD = 27
    ADJUST_FORWARD = 28
    CENTERING = 30          # Position adjustment between front/rear obstacles
    
    # Terminal States
    COMPLETED = 40          # Successfully parked
    PARKED = 40
    FAILED = 41             # Parking maneuver failed or aborted


class ParallelParkingController:
    """
    Controller executing automated parallel parking maneuver based on side & front distance sensors and odometry.
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        
        # State Tracking
        self.state = ParkingState.DISABLED
        self.prev_state = ParkingState.DISABLED
        self.enabled = False
        
        # Odometry Milestones
        self.state_start_time = 0.0
        self.spot_start_odom = 0.0
        self.spot_end_odom = 0.0
        self.spot_size = 0.0
        self.maneuver_start_odom = 0.0
        
        # Sensor Inputs
        self.left_distance = float('inf')
        self.front_distance = float('inf')
        self.right_distance = float('inf')
        self.current_odometry = 0.0
        
        # Control Output
        self.steering_output = 0
        self.speed_output = 0
        
        self._log("Parallel Parking Controller Initialized")

    def _log(self, msg):
        if self.logger:
            self.logger.info(f'[PARKING] {msg}')
        else:
            print(f'[PARKING] {msg}')

    def _change_state(self, new_state):
        if new_state != self.state:
            self.prev_state = self.state
            self.state = new_state
            self.state_start_time = time.time()
            self._log(f'State Transition: {self.prev_state.name} -> {self.state.name}')

    def enable(self):
        """Enable parallel parking controller"""
        if self.state == ParkingState.DISABLED:
            self.enabled = True
            self._change_state(ParkingState.IDLE)
            self._log("Parallel Parking Controller ENABLED")

    def disable(self):
        """Disable parallel parking controller"""
        self.enabled = False
        self._change_state(ParkingState.DISABLED)
        self.steering_output = 0
        self.speed_output = 0
        self._log("Parallel Parking Controller DISABLED")

    def update(self, sensors, odometry):
        """
        Execute FSM iteration.
        
        Args:
            sensors (list or tuple): Sensor readings [left, center/front, right/side]
            odometry (float): Total distance traveled in cm
            
        Returns:
            tuple: (steering_angle_cmd, speed_cmd)
        """
        if not self.enabled:
            return self._set_output(0, 0)
        
        # Process Sensor Values
        self.current_odometry = odometry
        no_detect_val = SensorConfig.SENSOR_NO_DETECTION_VALUE
        
        self.left_distance = sensors[0] if sensors[0] < no_detect_val else float('inf')
        self.front_distance = sensors[1] if sensors[1] < no_detect_val else float('inf')
        self.right_distance = sensors[2] if sensors[2] < no_detect_val else float('inf')
        
        # FSM Execution
        if self.state == ParkingState.DISABLED:
            return self._set_output(0, 0)
        
        elif self.state == ParkingState.IDLE:
            self.spot_start_odom = 0.0
            self.spot_size = 0.0
            self._change_state(ParkingState.SEARCHING_SPOT)
            return self._set_output(0, SpeedConfig.SEARCH_SPEED)
        
        elif self.state == ParkingState.SEARCHING_SPOT:
            return self._state_searching_spot()
        
        elif self.state == ParkingState.SPOT_DETECTED:
            return self._state_spot_detected()
        
        elif self.state == ParkingState.DRIVE_PAST_SPOT:
            return self._state_drive_past_spot()
        
        elif self.state == ParkingState.ALIGN_WITH_FRONT:
            return self._state_align_with_front()
        
        elif self.state == ParkingState.REVERSE_STAGE_1:
            return self._state_reverse_stage_1()
        
        elif self.state == ParkingState.REVERSE_STAGE_2:
            return self._state_reverse_stage_2()
        
        elif self.state == ParkingState.CENTERING:
            return self._state_centering()
        
        elif self.state in [ParkingState.PARKED, ParkingState.COMPLETED, ParkingState.FAILED]:
            return self._set_output(0, 0)
        
        return self._set_output(0, 0)

    def _state_searching_spot(self):
        """Drive forward scanning the right side sensor for an open parallel parking space"""
        is_space_open = (self.right_distance >= SensorConfig.SPACE_START_THRESHOLD or 
                         self.right_distance == float('inf'))
        
        if is_space_open:
            if self.spot_start_odom == 0.0:
                self.spot_start_odom = self.current_odometry
                self._log(f"Potential parking space started at odom: {self.spot_start_odom:.1f} cm")
            
            self.spot_size = self.current_odometry - self.spot_start_odom
        else:
            if self.spot_start_odom > 0.0:
                self.spot_end_odom = self.current_odometry
                self.spot_size = self.spot_end_odom - self.spot_start_odom
                self._log(f"Space ended. Total spot length: {self.spot_size:.1f} cm")
                
                if self.spot_size >= ParkingConfig.MIN_PARKING_SPACE:
                    self._log(f"✅ Valid spot found! Size: {self.spot_size:.1f} cm (Min required: {ParkingConfig.MIN_PARKING_SPACE} cm)")
                    self._change_state(ParkingState.SPOT_DETECTED)
                    return self._set_output(0, SpeedConfig.ALIGN_SPEED)
                else:
                    self._log(f"Spot too small ({self.spot_size:.1f} cm). Continuing search...")
                    self.spot_start_odom = 0.0
                    self.spot_size = 0.0
        
        if self.front_distance < SensorConfig.MIN_SAFE_DISTANCE:
            self._log("⚠️ Front obstacle warning during search. Stopping.")
            self._change_state(ParkingState.FAILED)
            return self._set_output(0, 0)
        
        return self._set_output(0, SpeedConfig.SEARCH_SPEED)

    def _state_spot_detected(self):
        self.maneuver_start_odom = self.current_odometry
        self._change_state(ParkingState.DRIVE_PAST_SPOT)
        return self._set_output(0, SpeedConfig.ALIGN_SPEED)

    def _state_drive_past_spot(self):
        distance_driven = self.current_odometry - self.maneuver_start_odom
        
        if distance_driven >= ParkingConfig.DRIVE_PAST_SPOT_DISTANCE:
            self._log(f"Drive past completed ({distance_driven:.1f} cm). Stopping for alignment.")
            self.maneuver_start_odom = self.current_odometry
            self._change_state(ParkingState.ALIGN_WITH_FRONT)
            return self._set_output(0, 0)
        
        return self._set_output(0, SpeedConfig.ALIGN_SPEED)

    def _state_align_with_front(self):
        elapsed = time.time() - self.state_start_time
        if elapsed >= 1.0:
            self._log("Starting Reverse Stage 1 (Turning sharp right into spot)...")
            self.maneuver_start_odom = self.current_odometry
            self._change_state(ParkingState.REVERSE_STAGE_1)
            return self._set_output(6, -SpeedConfig.PARKING_REVERSE_SPEED)
        
        return self._set_output(0, 0)

    def _state_reverse_stage_1(self):
        distance_reversed = abs(self.current_odometry - self.maneuver_start_odom)
        
        if distance_reversed >= ParkingConfig.REVERSE_STAGE1_DISTANCE:
            self._log(f"Reverse Stage 1 complete ({distance_reversed:.1f} cm). Switching to Reverse Stage 2...")
            self.maneuver_start_odom = self.current_odometry
            self._change_state(ParkingState.REVERSE_STAGE_2)
            return self._set_output(-6, -SpeedConfig.PARKING_REVERSE_SPEED)
        
        return self._set_output(6, -SpeedConfig.PARKING_REVERSE_SPEED)

    def _state_reverse_stage_2(self):
        distance_reversed = abs(self.current_odometry - self.maneuver_start_odom)
        
        if distance_reversed >= ParkingConfig.REVERSE_STAGE2_DISTANCE:
            self._log(f"Reverse Stage 2 complete ({distance_reversed:.1f} cm). Moving to forward centering...")
            self.maneuver_start_odom = self.current_odometry
            self._change_state(ParkingState.CENTERING)
            return self._set_output(0, SpeedConfig.CENTERING_SPEED)
        
        return self._set_output(-6, -SpeedConfig.PARKING_REVERSE_SPEED)

    def _state_centering(self):
        distance_centered = abs(self.current_odometry - self.maneuver_start_odom)
        
        if (distance_centered >= ParkingConfig.CENTERING_FORWARD_DISTANCE or 
            self.front_distance <= ParkingConfig.STOP_FRONT_TOLERANCE):
            self._log("=" * 60)
            self._log("🎉 PARALLEL PARKING MANEUVER COMPLETED SUCCESSFULLY!")
            self._log(f"Final front obstacle distance: {self.front_distance:.1f} cm")
            self._log("=" * 60)
            self._change_state(ParkingState.PARKED)
            return self._set_output(0, 0)
        
        return self._set_output(0, SpeedConfig.CENTERING_SPEED)

    def _set_output(self, steering, speed):
        self.steering_output = steering
        self.speed_output = speed
        return (steering, speed)

    def is_completed(self):
        return self.state in [ParkingState.PARKED, ParkingState.COMPLETED]

    def is_failed(self):
        return self.state == ParkingState.FAILED

    def get_status(self):
        return {
            'state': self.state.name,
            'enabled': self.enabled,
            'spot_size': self.spot_size,
            'front_dist': self.front_distance,
            'right_dist': self.right_distance,
            'left_dist': self.left_distance,
            'steering': self.steering_output,
            'speed': self.speed_output
        }


class DoubleParkingController:
    """
    Controller for parking between two obstacles (front and rear).
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        
        # Distance parameters
        self.SAFE_DISTANCE = 200.0
        self.SLOW_DISTANCE = 400.0
        self.STOP_DISTANCE = 120.0
        self.MIN_REAR_DISTANCE = 50.0
        self.IDEAL_CENTER_TOLERANCE = 30.0
        
        # Speeds
        self.DRIVE_SPEED = 6
        self.SLOW_SPEED = 4
        self.ADJUST_SPEED = 2
        self.SENSOR_NO_DETECTION = 1000.0
        
        # FSM State
        self.state = ParkingState.DISABLED
        self.prev_state = ParkingState.DISABLED
        
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        self.state_start_time = 0.0
        self.odometry = 0.0
        
        self.steering = 0
        self.speed_output = 0
        self.enabled = False
        
        self._log("Double Parking Controller Initialized")

    def _log(self, msg):
        if self.logger:
            self.logger.info(f'[DOUBLE PARKING] {msg}')
        else:
            print(f'[DOUBLE PARKING] {msg}')

    def _change_state(self, new_state):
        if new_state != self.state:
            self.prev_state = self.state
            self.state = new_state
            self.state_start_time = time.time()
            self._log(f'State: {self.prev_state.name} -> {self.state.name}')

    def enable(self):
        if self.state == ParkingState.DISABLED:
            self.enabled = True
            self._change_state(ParkingState.IDLE)
            self._log("Double Parking Controller ENABLED")

    def disable(self):
        self.enabled = False
        self._change_state(ParkingState.DISABLED)
        self.steering = 0
        self.speed_output = 0
        self._log("Double Parking Controller DISABLED")

    def update(self, sensors, odometry):
        self.left_distance = sensors[0] if sensors[0] < self.SENSOR_NO_DETECTION else float('inf')
        self.front_distance = sensors[1] if sensors[1] < self.SENSOR_NO_DETECTION else float('inf')
        self.right_distance = sensors[2] if sensors[2] < self.SENSOR_NO_DETECTION else float('inf')
        self.odometry = odometry
        
        if self.state == ParkingState.DISABLED:
            return self._stop()
        elif self.state == ParkingState.IDLE:
            self._change_state(ParkingState.DRIVE_FORWARD)
            return self._forward(self.DRIVE_SPEED)
        elif self.state == ParkingState.DRIVE_FORWARD:
            return self._state_drive_forward()
        elif self.state == ParkingState.SLOW_APPROACH:
            return self._state_slow_approach()
        elif self.state == ParkingState.STOPPING:
            return self._state_stopping()
        elif self.state == ParkingState.ADJUST_BACKWARD:
            return self._state_adjust_backward()
        elif self.state == ParkingState.ADJUST_FORWARD:
            return self._state_adjust_forward()
        elif self.state == ParkingState.CENTERING:
            return self._state_centering()
        elif self.state in [ParkingState.COMPLETED, ParkingState.PARKED, ParkingState.FAILED]:
            return self._stop()
        
        return self._stop()

    def _state_drive_forward(self):
        if self.front_distance < self.SLOW_DISTANCE:
            self._change_state(ParkingState.SLOW_APPROACH)
            return self._forward(self.SLOW_SPEED)
        return self._forward(self.DRIVE_SPEED)

    def _state_slow_approach(self):
        if self.front_distance <= self.STOP_DISTANCE:
            self._change_state(ParkingState.STOPPING)
            return self._stop()
        return self._forward(self.SLOW_SPEED)

    def _state_stopping(self):
        if time.time() - self.state_start_time < 1.0:
            return self._stop()
        self._change_state(ParkingState.CENTERING)
        return self._stop()

    def _state_adjust_backward(self):
        if time.time() - self.state_start_time < 2.0:
            return self._backward(self.ADJUST_SPEED)
        self._change_state(ParkingState.CENTERING)
        return self._stop()

    def _state_adjust_forward(self):
        if time.time() - self.state_start_time < 1.0:
            return self._forward(self.ADJUST_SPEED)
        self._change_state(ParkingState.CENTERING)
        return self._stop()

    def _state_centering(self):
        if self.front_distance < self.STOP_DISTANCE - 20:
            self._change_state(ParkingState.ADJUST_BACKWARD)
            return self._backward(self.ADJUST_SPEED)
        if self.front_distance > self.STOP_DISTANCE + 50:
            self._change_state(ParkingState.ADJUST_FORWARD)
            return self._forward(self.ADJUST_SPEED)
        
        self._change_state(ParkingState.COMPLETED)
        return self._stop()

    def _stop(self):
        self.steering = 0
        self.speed_output = 0
        return (0, 0)

    def _forward(self, speed, steering=0):
        self.steering = steering
        self.speed_output = speed
        return (steering, speed)

    def _backward(self, speed, steering=0):
        self.steering = steering
        self.speed_output = -speed
        return (steering, -speed)

    def is_completed(self):
        return self.state in [ParkingState.COMPLETED, ParkingState.PARKED]

    def is_failed(self):
        return self.state == ParkingState.FAILED

    def get_status(self):
        return {
            'state': self.state.name,
            'enabled': self.enabled,
            'front': self.front_distance,
            'left': self.left_distance,
            'right': self.right_distance,
            'steering': self.steering,
            'speed': self.speed_output
        }
