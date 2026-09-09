#!/usr/bin/env python3
"""
AVIS Engine Simulator Runner for Parallel Parking System
Connects to AVIS Engine simulator socket and executes parallel parking algorithm.
"""

import avisengine
import cv2
import numpy as np
import time
import sys

from config import AVISConfig, RobotConfig, SensorConfig
from parking_controller import ParallelParkingController, ParkingState


class Logger:
    """Console logger wrapper"""
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warn(self, msg):
        print(f"[WARN] {msg}")
    def error(self, msg):
        print(f"[ERROR] {msg}")


class AVISParallelParkingRobot:
    """
    Bridge connecting AVIS Engine Simulator to Parallel Parking FSM Controller.
    """
    
    def __init__(self):
        self.logger = Logger()
        
        # AVIS Connection
        self.car = None
        self.connected = False
        
        try:
            self.car = avisengine.Car()
            self.car.connect(AVISConfig.SERVER_IP, AVISConfig.SERVER_PORT)
            self.connected = True
            self.logger.info(f"✅ Connected to AVIS Engine Simulator at {AVISConfig.SERVER_IP}:{AVISConfig.SERVER_PORT}")
        except Exception as e:
            self.logger.error(f"Failed to connect to AVIS Engine: {e}")
            self.logger.error("Please ensure AVIS Engine simulator is running and TCP port 25001 is open!")
            sys.exit(1)
        
        # Parallel Parking Controller
        self.parking_controller = ParallelParkingController(self.logger)
        
        # Odometry Tracking
        self.odometry = 0.0
        self.last_time = time.time()
        
        # State Tracking
        self.state = 'starting'
        self.error_count = 0
        self.max_errors = 5

    def update_odometry(self):
        """Update odometry estimate based on vehicle velocity"""
        try:
            current_time = time.time()
            dt = current_time - self.last_time
            speed = abs(self.car.getSpeed())
            
            # Integrate speed over time to get distance in cm
            self.odometry += speed * dt * AVISConfig.ODOMETRY_SPEED_FACTOR * 10.0
            self.last_time = current_time
        except Exception:
            pass

    def reset_odometry(self):
        """Reset odometry counter"""
        self.odometry = 0.0
        self.last_time = time.time()

    def safe_stop(self):
        """Safe shutdown vehicle speed and steering"""
        try:
            if self.car:
                self.car.speed_value = 0
                self.car.steering_value = 0
                time.sleep(0.1)
        except Exception:
            pass

    def run(self):
        """Main execution loop"""
        self.logger.info("=" * 60)
        self.logger.info("🚗 Starting Parallel Parking System in AVIS Engine...")
        self.logger.info("Press 'Q' in OpenCV Window to stop execution.")
        self.logger.info("=" * 60)
        
        # Wait for simulator initial synchronization
        time.sleep(AVISConfig.PARKING_SEARCH_DELAY)
        
        # Enable controller
        self.state = 'parking'
        self.reset_odometry()
        self.parking_controller.enable()
        
        frame_count = 0
        
        while True:
            try:
                # 1. Fetch telemetry & sensor frame with timeout handling
                try:
                    self.car.getData()
                    self.error_count = 0  # Reset error counter on successful read
                except TimeoutError:
                    self.error_count += 1
                    self.logger.warn(f"Telemetry Timeout ({self.error_count}/{self.max_errors})")
                    if self.error_count >= self.max_errors:
                        self.logger.error("Connection lost - AVIS Engine may have closed")
                        break
                    time.sleep(0.2)
                    continue
                except Exception as e:
                    self.logger.error(f"getData error: {e}")
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        break
                    time.sleep(0.2)
                    continue
                
                # 2. Extract visual frame & sensors
                frame = self.car.getImage()
                raw_sensors = self.car.getSensors()
                
                if raw_sensors is None or len(raw_sensors) < 3:
                    time.sleep(0.05)
                    continue
                
                self.update_odometry()
                frame_count += 1
                
                # 3. Update Parallel Parking Controller FSM
                if self.state == 'parking':
                    steering_cmd, speed_cmd = self.parking_controller.update(
                        raw_sensors, self.odometry
                    )
                    
                    # Scale outputs for AVIS Engine
                    avis_steering = int(steering_cmd * AVISConfig.STEERING_MULTIPLIER)
                    avis_speed = int(speed_cmd * AVISConfig.SPEED_MULTIPLIER)
                    
                    # Clamp inputs to valid ranges
                    avis_steering = max(-30, min(30, avis_steering))
                    avis_speed = max(-30, min(30, avis_speed))
                    
                    # Apply commands to vehicle
                    self.car.steering_value = avis_steering
                    self.car.speed_value = avis_speed
                    
                    # Check completion status
                    if self.parking_controller.is_completed():
                        self.logger.info("=" * 60)
                        self.logger.info("🎉 PARALLEL PARKING COMPLETED SUCCESSFULLY!")
                        self.logger.info("=" * 60)
                        self.state = 'completed'
                    elif self.parking_controller.is_failed():
                        self.logger.warn("❌ PARALLEL PARKING FAILED!")
                        self.state = 'failed'
                
                elif self.state in ['completed', 'failed']:
                    self.safe_stop()
                    time.sleep(2)
                    break
                
                # 4. Display OpenCV Visual HUD Overlay
                if frame is not None:
                    try:
                        hud_frame = self.draw_hud_overlay(frame, raw_sensors)
                        cv2.imshow("AVIS Parallel Parking HUD", hud_frame)
                    except Exception as e:
                        self.logger.warn(f"Display overlay error: {e}")
                
                # 5. Keyboard input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("User requested stop via keyboard.")
                    break
                
                time.sleep(0.05)
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard Interrupt detected.")
                break
            except Exception as e:
                self.logger.error(f"Loop Exception: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        self.logger.info("Shutting down AVIS robot client...")
        cv2.destroyAllWindows()
        self.safe_stop()
        self.car = None
        self.logger.info("Shutdown complete.")

    def draw_hud_overlay(self, frame, sensors):
        """Render modern HUD overlay on video feed"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Semi-transparent top dark panel
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 260), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
        
        y = 28
        # Header Title
        cv2.putText(display, "AVIS PARALLEL PARKING SYSTEM", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        
        y += 32
        # FSM State & Status
        status = self.parking_controller.get_status()
        state_str = status['state']
        color_state = (0, 255, 0) if state_str == 'PARKED' else (0, 255, 255)
        
        cv2.putText(display, f"FSM State: {state_str}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_state, 2)
        
        y += 28
        # Vehicle Control Telemetry
        cv2.putText(display, f"Speed: {self.car.speed_value:3d}  | Steering: {self.car.steering_value:3d} deg", 
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        
        y += 28
        # Odometry & Spot Size
        cv2.putText(display, f"Odom: {self.odometry:6.1f} cm | Spot Size: {status['spot_size']:6.1f} cm", 
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 250, 200), 1)
        
        y += 32
        # Sensor Readings Display
        labels = ["Left", "Center (Front)", "Right (Side Scan)"]
        for i, label in enumerate(labels):
            val = sensors[i]
            val_str = f"{val:4.0f} cm" if val < SensorConfig.SENSOR_NO_DETECTION_VALUE else "CLEAR"
            
            if val < SensorConfig.MIN_SAFE_DISTANCE:
                color = (0, 0, 255)       # Red: Hazard
            elif val < SensorConfig.OBSTACLE_THRESHOLD:
                color = (0, 165, 255)     # Orange: Warning
            else:
                color = (0, 255, 100)     # Green: Clear
            
            cv2.putText(display, f"{label}: {val_str}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
            y += 26
        
        # Footer prompt
        cv2.putText(display, "Press 'Q' to Quit", (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║       🚗 AVIS Engine - Autonomous Parallel Parking           ║
║                                                              ║
║   Instructions:                                              ║
║     1. Launch AVIS Engine Simulator first                    ║
║     2. Ensure TCP port 25001 is listening                    ║
║     3. Run this script to execute parallel parking           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    robot = AVISParallelParkingRobot()
    robot.run()


if __name__ == '__main__':
    main()