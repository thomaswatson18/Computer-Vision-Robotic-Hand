import cv2
import mediapipe as mp
from picamera2 import Picamera2
import time
import board
import busio
from adafruit_pca9685 import PCA9685
import numpy as np
import math
import threading
from flask import Flask, render_template, Response

# --- 1. SETUP - FLASK WEB SERVER ---
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# --- 2. SETUP - AI BRAIN (MediaPipe) ---
print("Setting up MediaPipe Hands...")
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 3. SETUP - EYE (Camera) ---
print("Setting up Camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
print("Camera started.")

# --- 4. SETUP - NERVOUS SYSTEM & MUSCLES (Servo Driver) ---
print("Setting up Servo Driver...")
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50 
print("Servo Driver setup complete.")

# --- 5. HELPER FUNCTIONS ---

def set_servo_angle(channel, angle):
    angle = np.clip(angle, 0, 180)
    
    # --- FULL CALIBRATED RANGE RESTORED ---
    # We are using your exact measured limits for maximum range.
    pulse_min = 1900 
    pulse_max = 8700 
    
    duty_cycle = int(np.interp(angle, [0, 180], [pulse_min, pulse_max]))
    pca.channels[channel].duty_cycle = duty_cycle

def de_energize_servos():
    for i in range(5):
        pca.channels[i].duty_cycle = 0

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- 6. MAIN AI AND SERVO THREAD ---
def hand_tracking_thread():
    global output_frame, lock
    
    # Initial Values
    smooth_angles = [90.0] * 5
    last_sent_angles = [90.0] * 5
    
    # Dead Zone (Minimum change required to move)
    DEAD_ZONE = 2.0 
    
    print("Starting control loop in a separate thread...")
    for i in range(5):
        set_servo_angle(i, 90)
    time.sleep(1)

    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- FINGER CALCULATION ---
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            palm_scale = get_distance(wrist, middle_mcp)
            
            thumb_dist = get_distance(thumb_tip, middle_mcp)
            index_dist = get_distance(index_tip, index_mcp)
            middle_dist = get_distance(middle_tip, middle_mcp)
            ring_dist = get_distance(ring_tip, ring_mcp)
            pinky_dist = get_distance(pinky_tip, pinky_mcp)
            
            # Ratios
            thumb_ratio = thumb_dist / palm_scale
            index_ratio = index_dist / palm_scale
            middle_ratio = middle_dist / palm_scale
            ring_ratio = ring_dist / palm_scale
            pinky_ratio = pinky_dist / palm_scale
            
            # --- ANGLE MAPPING (Your Tuned Values) ---
            raw_thumb_angle = np.interp(thumb_ratio, [0.21, 0.73], [180, 0])
            raw_index_angle = np.interp(index_ratio, [0.17, 0.82], [180, 0])
            raw_middle_angle = np.interp(middle_ratio, [0.30, 0.88], [180, 0])
            raw_ring_angle = np.interp(ring_ratio, [0.30, 0.85], [180, 0])
            raw_pinky_angle = np.interp(pinky_ratio, [0.20, 0.70], [180, 0])

            raw_angles = [raw_thumb_angle, raw_index_angle, raw_middle_angle, raw_ring_angle, raw_pinky_angle]
            
            # --- ADAPTIVE SMOOTHING ---
            for i in range(5):
                # Calculate how much we need to move
                diff = abs(raw_angles[i] - smooth_angles[i])
                
                # If moving fast (big difference), be responsive (0.7)
                # If moving slow (small difference), be smooth (0.15)
                if diff > 10.0:
                    alpha = 0.7
                else:
                    alpha = 0.15
                
                # Apply smoothing
                smooth_angles[i] = (alpha * raw_angles[i]) + ((1.0 - alpha) * smooth_angles[i])
                
                # Move Servo (with small dead zone)
                new_angle = smooth_angles[i]
                if abs(new_angle - last_sent_angles[i]) > DEAD_ZONE:
                    set_servo_angle(i, new_angle)
                    last_sent_angles[i] = new_angle
            
            print(f"Angles: T={int(smooth_angles[0])} I={int(smooth_angles[1])} M={int(smooth_angles[2])} R={int(smooth_angles[3])} P={int(smooth_angles[4])}")
        
        else:
            print("Waiting for hand...")
            
        with lock:
            (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
            if not flag:
                continue
            output_frame = bytearray(encodedImage)
        
        time.sleep(0.04)

# --- 7. FLASK WEB SERVER FUNCTIONS ---

@app.route("/")
def index():
    return "Robot Hand Video Stream. Go to /video_feed"

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            frame_bytes = output_frame
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 8. START THE PROGRAM ---
if __name__ == '__main__':
    try:
        ai_thread = threading.Thread(target=hand_tracking_thread)
        ai_thread.daemon = True
        ai_thread.start()
        
        print("\n--- Web server starting! ---")
        print(f"Open this URL in a browser on your main computer:")
        print(f"http://192.168.1.193:8000/video_feed")
        
        app.run(host='0.0.0.0', port=8000, threaded=True)

    except KeyboardInterrupt:
        print("\nStopping program (Ctrl+C pressed)...")
    
    finally:
        de_energize_servos()
        picam2.stop()
        hands.close()
        pca.deinit()
        print("Cleanup complete. Servos de-energized. Camera off.")
