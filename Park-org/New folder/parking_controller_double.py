#!/usr/bin/env python3
"""
Double Parking Controller
Park between two obstacles (front and rear)

Uses FRONT CENTER sensor to detect obstacle ahead
"""

from enum import Enum
import time


class ParkingState(Enum):
    """Parking states"""
    
    DISABLED = 0
    IDLE = 1
    
    # Phase 1: Approach
    DRIVE_FORWARD = 10
    SLOW_APPROACH = 11
    
    # Phase 2: Stop & Adjust
    STOPPING = 20
    CHECK_REAR_DISTANCE = 21
    ADJUST_BACKWARD = 22
    ADJUST_FORWARD = 23
    
    # Phase 3: Final
    CENTERING = 30
    COMPLETED = 40
    FAILED = 41


class DoubleParkingController:
    """
    Controller for parking between two obstacles
    
    Scenario:
        [Obstacle 1] ---- [CAR] ---- [Obstacle 2]
           (rear)                      (front)
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        
        # ═══════════════════════════════════════════════════
        # پارامترهای قابل تنظیم
        # ═══════════════════════════════════════════════════
        
        # فاصله‌های مهم (واحد: سانتی‌متر)
        self.SAFE_DISTANCE = 200.0          # فاصله امن از مانع
        self.SLOW_DISTANCE = 400.0          # شروع کاهش سرعت
        self.STOP_DISTANCE = 120.0           # فاصله توقف از مانع جلو
        self.MIN_REAR_DISTANCE = 50.0       # حداقل فاصله از مانع عقب
        self.IDEAL_CENTER_TOLERANCE = 30.0  # تلرانس مرکزیت
        
        # سرعت‌ها
        self.DRIVE_SPEED = 6               # سرعت عادی
        self.SLOW_SPEED = 4                 # سرعت آهسته
        self.ADJUST_SPEED = 2               # سرعت تنظیم
        
        # سنسور
        self.SENSOR_NO_DETECTION = 1000.0   # مقدار سنسور وقتی چیزی نمی‌بینه
        
        # ═══════════════════════════════════════════════════
        # State
        # ═══════════════════════════════════════════════════
        
        self.state = ParkingState.DISABLED
        self.prev_state = ParkingState.DISABLED
        
        # Sensor data
        self.front_distance = float('inf')   # سنسور جلو (وسط)
        self.left_distance = float('inf')    # سنسور چپ
        self.right_distance = float('inf')   # سنسور راست
        
        # Tracking
        self.state_start_time = 0.0
        self.adjust_start_distance = 0.0
        
        # Odometry
        self.odometry = 0.0
        
        # Control output
        self.steering = 0
        self.speed_output = 0
        
        # Flags
        self.enabled = False
        
        self._log("Double Parking Controller initialized")

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
            self._log(f'State: {self.prev_state.name} -> {self.state.name}')

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════
    
    def enable(self):
        """فعال کردن پارک"""
        if self.state == ParkingState.DISABLED:
            self.enabled = True
            self._change_state(ParkingState.IDLE)
            self._log("Parking ENABLED")

    def disable(self):
        """غیرفعال کردن پارک"""
        self.enabled = False
        self._change_state(ParkingState.DISABLED)
        self.steering = 0
        self.speed_output = 0
        self._log("Parking DISABLED")

    def update(self, sensors, odometry):
        """
        به‌روزرسانی و دریافت فرمان
        
        Args:
            sensors: [left, center, right] از AVIS
            odometry: مسافت طی‌شده
            
        Returns:
            (steering, speed)
        """
        # پردازش سنسورها
        self.left_distance = sensors[0] if sensors[0] < self.SENSOR_NO_DETECTION else float('inf')
        self.front_distance = sensors[1] if sensors[1] < self.SENSOR_NO_DETECTION else float('inf')
        self.right_distance = sensors[2] if sensors[2] < self.SENSOR_NO_DETECTION else float('inf')
        self.odometry = odometry
        
        # State machine
        if self.state == ParkingState.DISABLED:
            return self._stop()
        
        elif self.state == ParkingState.IDLE:
            self._log("Starting double parking...")
            self._change_state(ParkingState.DRIVE_FORWARD)
            return self._forward(self.DRIVE_SPEED)
        
        elif self.state == ParkingState.DRIVE_FORWARD:
            return self._state_drive_forward()
        
        elif self.state == ParkingState.SLOW_APPROACH:
            return self._state_slow_approach()
        
        elif self.state == ParkingState.STOPPING:
            return self._state_stopping()
        
        elif self.state == ParkingState.CHECK_REAR_DISTANCE:
            return self._state_check_rear()
        
        elif self.state == ParkingState.ADJUST_BACKWARD:
            return self._state_adjust_backward()
        
        elif self.state == ParkingState.ADJUST_FORWARD:
            return self._state_adjust_forward()
        
        elif self.state == ParkingState.CENTERING:
            return self._state_centering()
        
        elif self.state == ParkingState.COMPLETED:
            return self._stop()
        
        elif self.state == ParkingState.FAILED:
            return self._stop()
        
        return self._stop()

    def is_completed(self):
        return self.state == ParkingState.COMPLETED

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

    # ═══════════════════════════════════════════════════════════════════
    # STATE HANDLERS
    # ═══════════════════════════════════════════════════════════════════
    
    def _state_drive_forward(self):
        """حرکت رو به جلو تا نزدیک شدن به مانع"""
        
        self._log(f"Front: {self.front_distance:.0f}cm")
        
        # اگر مانع جلو در فاصله کاهش سرعت
        if self.front_distance < self.SLOW_DISTANCE:
            self._log(f"Obstacle ahead at {self.front_distance:.0f}cm - slowing down")
            self._change_state(ParkingState.SLOW_APPROACH)
            return self._forward(self.SLOW_SPEED)
        
        # ادامه حرکت
        return self._forward(self.DRIVE_SPEED)

    def _state_slow_approach(self):
        """نزدیک شدن آهسته به مانع جلو"""
        
        self._log(f"Approaching... Front: {self.front_distance:.0f}cm")
        
        # اگر به فاصله توقف رسیدیم
        if self.front_distance <= self.STOP_DISTANCE:
            self._log(f"Stop distance reached: {self.front_distance:.0f}cm")
            self._change_state(ParkingState.STOPPING)
            return self._stop()
        
        # ادامه نزدیک شدن
        return self._forward(self.SLOW_SPEED)

    def _state_stopping(self):
        """توقف کامل"""
        
        self._log("Stopped. Checking position...")
        
        # کمی صبر کن
        if time.time() - self.state_start_time < 1.0:
            return self._stop()
        
        self._change_state(ParkingState.CENTERING)
        return self._stop()

    def _state_check_rear(self):
        """بررسی فاصله عقب (اختیاری)"""
        
        # این حالت برای وقتی است که سنسور عقب داشتیم
        # فعلاً skip می‌کنیم
        self._change_state(ParkingState.CENTERING)
        return self._stop()

    def _state_adjust_backward(self):
        """عقب رفتن برای تنظیم"""
        
        elapsed = time.time() - self.state_start_time
        
        # ۲ ثانیه عقب برو
        if elapsed < 2.0:
            return self._backward(self.ADJUST_SPEED)
        
        self._change_state(ParkingState.CENTERING)
        return self._stop()

    def _state_adjust_forward(self):
        """جلو رفتن برای تنظیم"""
        
        elapsed = time.time() - self.state_start_time
        
        # ۱ ثانیه جلو برو
        if elapsed < 1.0:
            return self._forward(self.ADJUST_SPEED)
        
        self._change_state(ParkingState.CENTERING)
        return self._stop()

    def _state_centering(self):
        """مرکزیت بین دو مانع"""
        
        self._log(f"Final position - Front: {self.front_distance:.0f}cm")
        
        # اگر خیلی نزدیک جلو هستیم
        if self.front_distance < self.STOP_DISTANCE - 20:
            self._log("Too close to front - adjusting backward")
            self._change_state(ParkingState.ADJUST_BACKWARD)
            return self._backward(self.ADJUST_SPEED)
        
        # اگر خیلی دور هستیم
        if self.front_distance > self.STOP_DISTANCE + 50:
            self._log("Too far from front - adjusting forward")
            self._change_state(ParkingState.ADJUST_FORWARD)
            return self._forward(self.ADJUST_SPEED)
        
        # موقعیت خوبه!
        self._log("=" * 50)
        self._log("PARKING COMPLETED!")
        self._log(f"Final distance from front: {self.front_distance:.0f}cm")
        self._log("=" * 50)
        self._change_state(ParkingState.COMPLETED)
        return self._stop()

    # ═══════════════════════════════════════════════════════════════════
    # CONTROL OUTPUTS
    # ═══════════════════════════════════════════════════════════════════
    
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