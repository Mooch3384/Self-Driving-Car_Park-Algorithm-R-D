#!/usr/bin/env python3
"""
Main Execution Script for Autonomous Vehicle Parking Simulator
Launches the Graphical Interactive 2D Simulator Environment.
"""

from parking_simulator import AutonomousParkingSimulator


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║       🚗 AUTONOMOUS VEHICLE PARALLEL & DOUBLE PARKING SIMULATOR           ║
║                                                                          ║
║   Key Controls & Features:                                               ║
║     • [Space] : Toggle between AUTO (FSM Algorithm) and MANUAL Driving   ║
║     • [S]     : Switch Scenario Preset (PARALLEL / DOUBLE Parking)       ║
║     • [R]     : Reset Vehicle and Odometry Telemetry                      ║
║     • [P]     : Pause / Resume Simulation Execution                      ║
║     • Mouse   : Click & Drag Obstacles/Cars anywhere in real-time        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    sim = AutonomousParkingSimulator(width=1280, height=720)
    sim.run()


if __name__ == '__main__':
    main()
