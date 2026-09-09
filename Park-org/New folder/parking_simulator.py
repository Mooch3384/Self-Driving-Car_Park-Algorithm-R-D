#!/usr/bin/env python3
"""
Autonomous Vehicle Parallel & Double Parking Simulator
Interactive 2D Simulator with Ackermann Kinematics, Raycasting Ultrasonic Sensors,
Interactive Obstacle Dragging, and Modern Dark Glassmorphic Visual HUD.
"""

import sys
import math
import time
import pygame

from config import RobotConfig, SensorConfig, ParkingConfig, SpeedConfig
from parking_controller import ParallelParkingController, DoubleParkingController, ParkingState


# ═══════════════════════════════════════════════════════════════════════════
# COLOR PALETTE & DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
COLOR_BG = (18, 22, 28)
COLOR_ASPHALT = (28, 34, 43)
COLOR_ROAD_LINE = (200, 210, 225)
COLOR_CURB = (100, 110, 125)

# Vehicles
COLOR_EGO_BODY = (0, 200, 120)        # Emerald Green Ego Car
COLOR_EGO_WHEEL = (30, 30, 30)
COLOR_EGO_GLASS = (150, 230, 255, 180)
COLOR_PARKED_BODY1 = (70, 90, 120)    # Slate Blue Parked Car
COLOR_PARKED_BODY2 = (140, 70, 90)    # Crimson Parked Car

# Sensors & UI
COLOR_SENSOR_CLEAR = (0, 230, 118, 160)     # Neon Green
COLOR_SENSOR_WARN = (255, 214, 0, 180)      # Neon Yellow
COLOR_SENSOR_DANGER = (255, 23, 68, 220)    # Neon Red
COLOR_TRAJECTORY = (0, 180, 255, 120)

COLOR_PANEL_BG = (25, 32, 42, 220)
COLOR_PANEL_BORDER = (45, 58, 75)
COLOR_TEXT_PRIMARY = (240, 245, 250)
COLOR_TEXT_MUTED = (140, 155, 175)
COLOR_ACCENT = (0, 220, 255)


# ═══════════════════════════════════════════════════════════════════════════
# OBSTACLE & VEHICLE KINEMATICS CLASS
# ═══════════════════════════════════════════════════════════════════════════

class ObstacleRect:
    """Obstacle bounding box representing parked cars or obstacles"""
    def __init__(self, x, y, width, height, color=COLOR_PARKED_BODY1, label="Obstacle"):
        self.x = x          # Top-left x (cm)
        self.y = y          # Top-left y (cm)
        self.width = width  # (cm)
        self.height = height# (cm)
        self.color = color
        self.label = label
        self.is_dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def contains_point(self, px, py):
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)


class SimulatorCar:
    """Ackermann Kinematic Bicycle Model for Autonomous Vehicle"""
    def __init__(self, x=100.0, y=200.0, yaw_deg=0.0):
        self.x = x                  # Center x position (cm)
        self.y = y                  # Center y position (cm)
        self.yaw = math.radians(yaw_deg)  # Heading angle (rad)
        self.speed = 0.0            # Linear velocity (cm/s)
        self.steering = 0.0         # Steering angle (rad)
        
        # Dimensions (cm)
        self.length = RobotConfig.ROBOT_LENGTH    # 400 cm
        self.width = RobotConfig.ROBOT_WIDTH      # 180 cm
        self.wheelbase = RobotConfig.WHEELBASE    # 250 cm
        
        # Odometry tracking
        self.total_odometry = 0.0
        self.trajectory_history = []

    def reset(self, x=100.0, y=200.0, yaw_deg=0.0):
        self.x = x
        self.y = y
        self.yaw = math.radians(yaw_deg)
        self.speed = 0.0
        self.steering = 0.0
        self.total_odometry = 0.0
        self.trajectory_history.clear()

    def update_physics(self, dt):
        """Bicycle Kinematic Model Integration"""
        if abs(self.speed) > 1e-4:
            # Kinematic equations
            dx = self.speed * math.cos(self.yaw) * dt
            dy = self.speed * math.sin(self.yaw) * dt
            dyaw = (self.speed / self.wheelbase) * math.tan(self.steering) * dt
            
            self.x += dx
            self.y += dy
            self.yaw += dyaw
            
            # Odometry accumulator
            ds = math.hypot(dx, dy)
            self.total_odometry += ds
            
            # Record trajectory points
            if not self.trajectory_history or math.hypot(self.x - self.trajectory_history[-1][0], 
                                                         self.y - self.trajectory_history[-1][1]) > 10:
                self.trajectory_history.append((self.x, self.y))

    def get_corners(self):
        """Calculate bounding box corners in world coordinates"""
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        
        l_half = self.length / 2.0
        w_half = self.width / 2.0
        
        # Local corner coordinates relative to vehicle center
        local_corners = [
            (+l_half, +w_half),
            (+l_half, -w_half),
            (-l_half, -w_half),
            (-l_half, +w_half)
        ]
        
        world_corners = []
        for lx, ly in local_corners:
            wx = self.x + (lx * cos_y - ly * sin_y)
            wy = self.y + (lx * sin_y + ly * cos_y)
            world_corners.append((wx, wy))
            
        return world_corners


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousParkingSimulator:
    """Interactive Graphical Simulator for Parallel & Double Parking Systems"""
    
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Autonomous Vehicle Parallel & Double Parking Simulator")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.SysFont("Inter", 22, bold=True)
        self.font_main = pygame.font.SysFont("Inter", 16, bold=True)
        self.font_small = pygame.font.SysFont("Inter", 13)
        
        # Vehicle & Controllers
        self.car = SimulatorCar(x=120.0, y=200.0, yaw_deg=0.0)
        self.parallel_controller = ParallelParkingController()
        self.double_controller = DoubleParkingController()
        
        # Simulator Settings
        self.scenario = "PARALLEL"   # "PARALLEL" or "DOUBLE"
        self.mode = "AUTO"           # "AUTO" or "MANUAL"
        self.paused = False
        self.active_controller = self.parallel_controller
        self.active_controller.enable()
        
        # Dragging state
        self.selected_obstacle = None
        self.obstacles = []
        self.setup_scenario(self.scenario)
        
    def setup_scenario(self, scenario_name):
        """Setup scenario obstacles and initial car position"""
        self.scenario = scenario_name
        self.obstacles.clear()
        
        if scenario_name == "PARALLEL":
            # Road curb boundary on right side
            self.obstacles.append(ObstacleRect(x=0, y=420, width=1280, height=40, color=COLOR_CURB, label="Curb"))
            
            # Rear Parked Car (x=200, y=280)
            self.obstacles.append(ObstacleRect(x=200, y=280, width=380, height=130, color=COLOR_PARKED_BODY1, label="Rear Car"))
            
            # Front Parked Car (x=1200 - 380 = 820, y=280) -> Gap = 820 - (200+380) = 240 + 380 = 620 cm (Ideal spot)
            self.obstacles.append(ObstacleRect(x=820, y=280, width=380, height=130, color=COLOR_PARKED_BODY2, label="Front Car"))
            
            # Ego Car Start Position
            self.car.reset(x=100.0, y=200.0, yaw_deg=0.0)
            self.active_controller = self.parallel_controller
            self.parallel_controller.disable()
            self.parallel_controller.enable()
            
        elif scenario_name == "DOUBLE":
            # Front obstacle ahead
            self.obstacles.append(ObstacleRect(x=600, y=140, width=300, height=140, color=COLOR_PARKED_BODY2, label="Front Obstacle"))
            # Rear obstacle behind
            self.obstacles.append(ObstacleRect(x=50, y=140, width=200, height=140, color=COLOR_PARKED_BODY1, label="Rear Obstacle"))
            
            self.car.reset(x=280.0, y=200.0, yaw_deg=0.0)
            self.active_controller = self.double_controller
            self.double_controller.disable()
            self.double_controller.enable()

    # ═══════════════════════════════════════════════════════════════════════
    # RAYCASTING ULTRASONIC SENSOR ENGINE
    # ═══════════════════════════════════════════════════════════════════════

    def raycast_sensor(self, origin_x, origin_y, angle_rad, max_range=1000.0):
        """Cast ray from (origin_x, origin_y) in direction angle_rad against obstacles"""
        min_distance = max_range
        hit_point = (origin_x + max_range * math.cos(angle_rad), 
                     origin_y + max_range * math.sin(angle_rad))
        
        # Step through obstacles
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        for obs in self.obstacles:
            r = obs.get_rect()
            # Check line intersection with box edges
            lines = [
                ((r.left, r.top), (r.right, r.top)),
                ((r.right, r.top), (r.right, r.bottom)),
                ((r.right, r.bottom), (r.left, r.bottom)),
                ((r.left, r.bottom), (r.left, r.top))
            ]
            
            for (p1x, p1y), (p2x, p2y) in lines:
                # Ray line segment vs line segment intersection
                dist, ix, iy = self._line_intersection(origin_x, origin_y, 
                                                        origin_x + max_range * cos_a, 
                                                        origin_y + max_range * sin_a, 
                                                        p1x, p1y, p2x, p2y)
                if dist is not None and dist < min_distance:
                    min_distance = dist
                    hit_point = (ix, iy)
                    
        return min_distance, hit_point

    def _line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None, None, None
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            ix = x1 + ua * (x2 - x1)
            iy = y1 + ua * (y2 - y1)
            dist = math.hypot(ix - x1, iy - y1)
            return dist, ix, iy
        return None, None, None

    def get_sensors_telemetry(self):
        """
        Calculate 3 sensor distance readings:
        Index 0: Left Sensor (-90 deg)
        Index 1: Front/Center Sensor (0 deg)
        Index 2: Right/Side Sensor (+90 deg)
        """
        cos_y = math.cos(self.car.yaw)
        sin_y = math.sin(self.car.yaw)
        l_half = self.car.length / 2.0
        w_half = self.car.width / 2.0
        
        # Sensor Origin Locations on vehicle body
        # Left sensor: Left flank center
        left_orig_x = self.car.x - w_half * sin_y
        left_orig_y = self.car.y + w_half * cos_y
        left_angle = self.car.yaw - math.pi / 2.0
        
        # Front sensor: Front bumper center
        front_orig_x = self.car.x + l_half * cos_y
        front_orig_y = self.car.y + l_half * sin_y
        front_angle = self.car.yaw
        
        # Right sensor: Right flank center
        right_orig_x = self.car.x + w_half * sin_y
        right_orig_y = self.car.y - w_half * cos_y
        right_angle = self.car.yaw + math.pi / 2.0
        
        d_left, pt_left = self.raycast_sensor(left_orig_x, left_orig_y, left_angle)
        d_front, pt_front = self.raycast_sensor(front_orig_x, front_orig_y, front_angle)
        d_right, pt_right = self.raycast_sensor(right_orig_x, right_orig_y, right_angle)
        
        sensor_data = [d_left, d_front, d_right]
        sensor_rays = [
            ((left_orig_x, left_orig_y), pt_left, d_left),
            ((front_orig_x, front_orig_y), pt_front, d_front),
            ((right_orig_x, right_orig_y), pt_right, d_right)
        ]
        
        return sensor_data, sensor_rays

    # ═══════════════════════════════════════════════════════════════════
    # SIMULATION LOOP & RENDERING
    # ═══════════════════════════════════════════════════════════════════

    def run(self):
        running = True
        
        while running:
            dt = self.clock.tick(60) / 1000.0  # Seconds per frame (60 FPS)
            
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.mode = "MANUAL" if self.mode == "AUTO" else "AUTO"
                    elif event.key == pygame.K_s:
                        next_sc = "DOUBLE" if self.scenario == "PARALLEL" else "PARALLEL"
                        self.setup_scenario(next_sc)
                    elif event.key == pygame.K_r:
                        self.setup_scenario(self.scenario)
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                        
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mx, my = event.pos
                        for obs in self.obstacles:
                            if obs.contains_point(mx, my):
                                self.selected_obstacle = obs
                                obs.is_dragging = True
                                obs.drag_offset_x = mx - obs.x
                                obs.drag_offset_y = my - obs.y
                                break
                                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.selected_obstacle:
                        self.selected_obstacle.is_dragging = False
                        self.selected_obstacle = None
                        
                elif event.type == pygame.MOUSEMOTION:
                    if self.selected_obstacle and self.selected_obstacle.is_dragging:
                        mx, my = event.pos
                        self.selected_obstacle.x = mx - self.selected_obstacle.drag_offset_x
                        self.selected_obstacle.y = my - self.selected_obstacle.drag_offset_y

            # 2. Physics & Control Logic Update
            if not self.paused:
                sensor_data, sensor_rays = self.get_sensors_telemetry()
                
                if self.mode == "AUTO":
                    # Update FSM Controller
                    steering_cmd, speed_cmd = self.active_controller.update(
                        sensor_data, self.car.total_odometry
                    )
                    
                    # Convert controller speed & steering to car physics parameters
                    self.car.speed = speed_cmd * 12.0  # Scale speed for visual smoothness
                    
                    # Controller steering range (-6 to +6) to angle in radians
                    max_steer_rad = math.radians(RobotConfig.MAX_STEERING_ANGLE)
                    self.car.steering = (steering_cmd / 6.0) * max_steer_rad
                    
                else:  # MANUAL MODE
                    keys = pygame.key.get_pressed()
                    speed_val = 0.0
                    steer_val = 0.0
                    
                    if keys[pygame.K_UP] or keys[pygame.K_w]:
                        speed_val = 80.0
                    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                        speed_val = -60.0
                        
                    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                        steer_val = -math.radians(25)
                    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                        steer_val = math.radians(25)
                        
                    self.car.speed = speed_val
                    self.car.steering = steer_val
                
                self.car.update_physics(dt)
            else:
                sensor_data, sensor_rays = self.get_sensors_telemetry()

            # 3. Graphical Rendering
            self.render(sensor_data, sensor_rays)
            
        pygame.quit()
        sys.exit(0)

    # ═══════════════════════════════════════════════════════════════════
    # DRAWING & GRAPHICS ENGINE
    # ═══════════════════════════════════════════════════════════════════

    def render(self, sensor_data, sensor_rays):
        self.screen.fill(COLOR_BG)
        
        # 1. Draw Road Environment
        pygame.draw.rect(self.screen, COLOR_ASPHALT, (0, 80, self.width, 360))
        # Dashed lane markings
        for lx in range(0, self.width, 40):
            pygame.draw.line(self.screen, COLOR_ROAD_LINE, (lx, 260), (lx + 20, 260), 2)
            
        # 2. Draw Trajectory History Trail
        if len(self.car.trajectory_history) > 1:
            pts = [(int(tx), int(ty)) for tx, ty in self.car.trajectory_history]
            pygame.draw.lines(self.screen, (0, 180, 255), False, pts, 2)

        # 3. Draw Obstacles
        for obs in self.obstacles:
            r = obs.get_rect()
            pygame.draw.rect(self.screen, obs.color, r, border_radius=6)
            pygame.draw.rect(self.screen, (200, 210, 220), r, width=2, border_radius=6)
            # Label
            lbl = self.font_small.render(obs.label, True, COLOR_TEXT_PRIMARY)
            self.screen.blit(lbl, (obs.x + 8, obs.y + 8))

        # 4. Draw Sensor Beams
        for (orig, hit, dist) in sensor_rays:
            if dist < SensorConfig.MIN_SAFE_DISTANCE:
                beam_color = COLOR_SENSOR_DANGER
            elif dist < SensorConfig.OBSTACLE_THRESHOLD:
                beam_color = COLOR_SENSOR_WARN
            else:
                beam_color = COLOR_SENSOR_CLEAR
                
            pygame.draw.line(self.screen, beam_color[:3], (int(orig[0]), int(orig[1])), (int(hit[0]), int(hit[1])), 2)
            pygame.draw.circle(self.screen, beam_color[:3], (int(hit[0]), int(hit[1])), 4)

        # 5. Draw Ego Vehicle
        self.draw_car(self.car)

        # 6. Draw Top HUD Visual Panel
        self.draw_hud_panel(sensor_data)

        pygame.display.flip()

    def draw_car(self, car):
        """Render Ego Autonomous Vehicle with rotation and wheels"""
        l_half = car.length / 2.0
        w_half = car.width / 2.0
        
        # Vehicle Surface
        car_surf = pygame.Surface((car.length, car.width), pygame.SRCALPHA)
        
        # Chassis Body
        pygame.draw.rect(car_surf, COLOR_EGO_BODY, (0, 0, car.length, car.width), border_radius=10)
        pygame.draw.rect(car_surf, (255, 255, 255), (0, 0, car.length, car.width), width=2, border_radius=10)
        
        # Windshield & Roof
        pygame.draw.rect(car_surf, COLOR_EGO_GLASS, (car.length * 0.3, 10, car.length * 0.4, car.width - 20), border_radius=4)
        
        # Front Headlights
        pygame.draw.circle(car_surf, (255, 255, 200), (int(car.length - 8), 18), 6)
        pygame.draw.circle(car_surf, (255, 255, 200), (int(car.length - 8), int(car.width - 18)), 6)
        
        # Rotate Vehicle Surface by Heading Angle
        angle_deg = -math.degrees(car.yaw)
        rotated_surf = pygame.transform.rotate(car_surf, angle_deg)
        new_rect = rotated_surf.get_rect(center=(int(car.x), int(car.y)))
        
        self.screen.blit(rotated_surf, new_rect.topleft)

    def draw_hud_panel(self, sensor_data):
        """Render Top Dark Glassmorphic HUD Dashboard Panel"""
        panel_rect = pygame.Rect(15, 15, self.width - 30, 95)
        
        # Glassmorphism container
        panel_surf = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, COLOR_PANEL_BG, (0, 0, panel_rect.width, panel_rect.height), border_radius=12)
        pygame.draw.rect(panel_surf, COLOR_PANEL_BORDER, (0, 0, panel_rect.width, panel_rect.height), width=2, border_radius=12)
        self.screen.blit(panel_surf, panel_rect.topleft)
        
        # 1. Title & Mode Status
        txt_title = self.font_title.render("AUTONOMOUS PARKING SIMULATOR", True, COLOR_ACCENT)
        self.screen.blit(txt_title, (30, 25))
        
        mode_str = f"MODE: {self.mode} [Space] | SCENARIO: {self.scenario} [S]"
        txt_mode = self.font_small.render(mode_str, True, COLOR_TEXT_MUTED)
        self.screen.blit(txt_mode, (30, 55))
        
        # 2. Controller FSM State Badge
        status = self.active_controller.get_status()
        state_name = status['state']
        
        badge_color = (0, 230, 118) if state_name == "PARKED" else (0, 220, 255)
        txt_fsm = self.font_main.render(f"FSM State: {state_name}", True, badge_color)
        self.screen.blit(txt_fsm, (440, 25))
        
        txt_odom = self.font_small.render(f"Odometry: {self.car.total_odometry:6.1f} cm | Spot: {status.get('spot_size', 0.0):6.1f} cm", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(txt_odom, (440, 55))
        
        # 3. Telemetry Indicators (Speed & Steering Angle)
        steer_deg = math.degrees(self.car.steering)
        txt_telemetry = self.font_small.render(f"Speed: {self.car.speed:5.1f} cm/s | Steering: {steer_deg:4.1f}°", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(txt_telemetry, (740, 25))
        
        # 4. Sensor Readings Dashboard Box
        s_left, s_front, s_right = sensor_data
        txt_sensors = self.font_small.render(f"Sensors -> L: {s_left:4.0f}cm | C(Front): {s_front:4.0f}cm | R(Side): {s_right:4.0f}cm", True, COLOR_ACCENT)
        self.screen.blit(txt_sensors, (740, 55))
        
        # Controls key hint on bottom right of screen
        hint_str = "Controls: [Space] Toggle Auto/Manual | [S] Scenario | [R] Reset | [P] Pause | Mouse: Drag Obstacles"
        txt_hint = self.font_small.render(hint_str, True, COLOR_TEXT_MUTED)
        self.screen.blit(txt_hint, (self.width - 640, self.height - 25))


def main():
    sim = AutonomousParkingSimulator()
    sim.run()


if __name__ == '__main__':
    main()
