/*
 * ESP32 MicroROS Node: Ultrasonic Sensor + Encoder Telemetry + Motor/Servo Control
 * 
 * Topics Published:
 *   /ultrasonic/distance (Float32) - Distance in cm
 *   /encoder_pulses (Int32) - Cumulative encoder pulse count
 * 
 * Topics Subscribed:
 *   /servo (Int8) - Steering command (-6 to +6)
 *   /motor_speed (Int8) - Speed command (-15 to +15)
 * 
 * Hardware:
 *   - HC-SR04 Ultrasonic Sensor (Trig: 5, Echo: 18)
 *   - Optical Quadrature Encoder (Pin A: 19, Pin B: 21)
 *   - Motor Driver (PWM: 22, Dir1: 23, Dir2: 25)
 *   - Steering Servo (Pin 26)
 */

#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/int8.h>

// ═══════════════════════════════════════════════════════════════════════════
// PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

#define ULTRASONIC_TRIG     5
#define ULTRASONIC_ECHO     18

#define ENCODER_PIN_A       19
#define ENCODER_PIN_B       21

#define MOTOR_PWM           22
#define MOTOR_DIR1          23
#define MOTOR_DIR2          25

#define SERVO_PIN           26
#define LED_PIN             2

// ═══════════════════════════════════════════════════════════════════════════
// ROS OBJECTS & MESSAGES
// ═══════════════════════════════════════════════════════════════════════════

rcl_publisher_t ultrasonic_pub;
rcl_publisher_t encoder_pub;
rcl_subscription_t servo_sub;
rcl_subscription_t motor_sub;

std_msgs__msg__Float32 ultrasonic_msg;
std_msgs__msg__Int32 encoder_msg;
std_msgs__msg__Int8 servo_msg;
std_msgs__msg__Int8 motor_msg;

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

// ═══════════════════════════════════════════════════════════════════════════
// ENCODER & HARDWARE VARIABLES
// ═══════════════════════════════════════════════════════════════════════════

volatile long encoder_pulses = 0;
volatile int last_a = 0;
volatile int last_b = 0;

// ═══════════════════════════════════════════════════════════════════════════
// INTERRUPT HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

void IRAM_ATTR encoder_isr_a() {
    int a = digitalRead(ENCODER_PIN_A);
    int b = digitalRead(ENCODER_PIN_B);
    
    if (a != last_a) {
        if (a == b) {
            encoder_pulses++;
        } else {
            encoder_pulses--;
        }
        last_a = a;
    }
}

void IRAM_ATTR encoder_isr_b() {
    int a = digitalRead(ENCODER_PIN_A);
    int b = digitalRead(ENCODER_PIN_B);
    
    if (b != last_b) {
        if (a != b) {
            encoder_pulses++;
        } else {
            encoder_pulses--;
        }
        last_b = b;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ULTRASONIC SENSOR (OPTIMIZED NON-BLOCKING READ)
// ═══════════════════════════════════════════════════════════════════════════

float read_ultrasonic() {
    digitalWrite(ULTRASONIC_TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(ULTRASONIC_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(ULTRASONIC_TRIG, LOW);
    
    // Non-blocking timeout capped at 12000 microseconds (~200 cm max range)
    long duration = pulseIn(ULTRASONIC_ECHO, HIGH, 12000);
    
    if (duration == 0) {
        return 999.0f;  // No reflection detected
    }
    
    // Distance (cm) = (Duration * 0.0343) / 2
    float distance = (duration * 0.0343f) / 2.0f;
    return distance;
}

// ═══════════════════════════════════════════════════════════════════════════
// MOTOR & SERVO DRIVERS
// ═══════════════════════════════════════════════════════════════════════════

void set_motor(int speed) {
    // Speed range: -15 to +15 mapped to PWM 0-255
    if (speed > 0) {
        digitalWrite(MOTOR_DIR1, HIGH);
        digitalWrite(MOTOR_DIR2, LOW);
        analogWrite(MOTOR_PWM, min(speed * 17, 255));
    } else if (speed < 0) {
        digitalWrite(MOTOR_DIR1, LOW);
        digitalWrite(MOTOR_DIR2, HIGH);
        analogWrite(MOTOR_PWM, min(-speed * 17, 255));
    } else {
        digitalWrite(MOTOR_DIR1, LOW);
        digitalWrite(MOTOR_DIR2, LOW);
        analogWrite(MOTOR_PWM, 0);
    }
}

void set_servo(int angle) {
    // Steering range: -6 to +6 mapped to PWM pulse (1000us - 2000us)
    int pulse_width = map(angle, -6, 6, 1000, 2000);
    ledcWrite(0, pulse_width);
}

// ═══════════════════════════════════════════════════════════════════════════
// ROS CALLBACKS
// ═══════════════════════════════════════════════════════════════════════════

void timer_callback(rcl_timer_t *timer, int64_t last_call_time) {
    RCLC_UNUSED(last_call_time);
    
    if (timer != NULL) {
        ultrasonic_msg.data = read_ultrasonic();
        rcl_publish(&ultrasonic_pub, &ultrasonic_msg, NULL);
        
        encoder_msg.data = encoder_pulses;
        rcl_publish(&encoder_pub, &encoder_msg, NULL);
        
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    }
}

void servo_callback(const void *msg_in) {
    const std_msgs__msg__Int8 *msg = (const std_msgs__msg__Int8 *)msg_in;
    set_servo(msg->data);
}

void motor_callback(const void *msg_in) {
    const std_msgs__msg__Int8 *msg = (const std_msgs__msg__Int8 *)msg_in;
    set_motor(msg->data);
}

// ═══════════════════════════════════════════════════════════════════════════
// SETUP & MAIN LOOP
// ═══════════════════════════════════════════════════════════════════════════

void setup() {
    pinMode(ULTRASONIC_TRIG, OUTPUT);
    pinMode(ULTRASONIC_ECHO, INPUT);
    pinMode(ENCODER_PIN_A, INPUT_PULLUP);
    pinMode(ENCODER_PIN_B, INPUT_PULLUP);
    pinMode(MOTOR_PWM, OUTPUT);
    pinMode(MOTOR_DIR1, OUTPUT);
    pinMode(MOTOR_DIR2, OUTPUT);
    pinMode(LED_PIN, OUTPUT);
    
    ledcSetup(0, 50, 16);  // PWM channel 0, 50Hz, 16-bit
    ledcAttachPin(SERVO_PIN, 0);
    
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_A), encoder_isr_a, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_B), encoder_isr_b, CHANGE);
    
    set_motor(0);
    set_servo(0);
    
    set_microros_transports();
    delay(2000);
    
    allocator = rcl_get_default_allocator();
    rclc_support_init(&support, 0, NULL, &allocator);
    rclc_node_init_default(&node, "esp32_node", "", &support);
    
    rclc_publisher_init_default(
        &ultrasonic_pub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
        "/ultrasonic/distance"
    );
    
    rclc_publisher_init_default(
        &encoder_pub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
        "/encoder_pulses"
    );
    
    rclc_subscription_init_default(
        &servo_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int8),
        "/servo"
    );
    
    rclc_subscription_init_default(
        &motor_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int8),
        "/motor_speed"
    );
    
    rclc_timer_init_default(
        &timer, &support, RCL_MS_TO_NS(50), timer_callback
    );
    
    rclc_executor_init(&executor, &support.context, 3, &allocator);
    rclc_executor_add_timer(&executor, &timer);
    rclc_executor_add_subscription(&executor, &servo_sub, &servo_msg, &servo_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &motor_sub, &motor_msg, &motor_callback, ON_NEW_DATA);
}

void loop() {
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));
}