from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import mediapipe as mp
import cv2 as cv
import time
import csv
import threading
from queue import Queue
from utils import *
from model import KeyPointClassifier

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

cap = None
frame_queue = Queue(maxsize=10)
capture_event = threading.Event()  # Event to control the frame capture thread
frame_count = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

keypoints_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

@app.route('/')
def index():
    return render_template('index.html')

def process_frame(frame):
    hand_sign = "NOT DETECTED"
    image = cv.flip(frame, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_process_landmark_list = pre_process_landmark(landmark_list)
            hand_sign_id = keypoints_classifier(pre_process_landmark_list)
            hand_sign = keypoint_classifier_labels[hand_sign_id]
            image = draw_landmarks(image, landmark_list)
    
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return hand_sign, image

def frame_capture():
    global cap
    while capture_event.is_set():  # Continue running while the event is set
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if not frame_queue.full():
                    frame_queue.put(frame)
                else:
                    print("Frame queue is full, dropping frames.")
            else:
                print("Error: Could not read frame.")
        else:
            print("Camera is not initialized or opened.")

@socketio.on('request_frame')
def stream_frame():
    global frame_count
    skip_frames = 2  # Process every nth frame (adjust as needed)

    if not frame_queue.empty():
        frame = frame_queue.get()
        
        # Record the start time for FPS calculation
        start_tick = cv.getTickCount()

        # Only process every nth frame
        if frame_count == 0:
            hand_sign, processed_frame = process_frame(frame)

            _, buffer = cv.imencode('.jpg', processed_frame)
            frame_data = buffer.tobytes()

            # Calculate FPS
            end_tick = cv.getTickCount()
            elapsed_time = (end_tick - start_tick) / cv.getTickFrequency()
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0

            emit('video_frame', frame_data)
            emit('info', hand_sign)
            emit('fps', round(fps, 2))  # Emit FPS rounded to two decimal places

        frame_count += 1  # Increment frame_count for each frame received
        frame_count %= skip_frames
    else:
        print("Frame queue is empty.")

    # Limit the frame rate
    socketio.sleep(1 / 15)

@socketio.on('connect')
def start_video_stream():
    global cap
    if cap is None:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        capture_event.set()  # Set the event to start the frame capture
        threading.Thread(target=frame_capture, daemon=True).start()

@socketio.on('disconnect')
def handle_disconnect():
    global cap
    if cap is not None:
        capture_event.clear()  # Clear the event to stop the frame capture loop
        cap.release()
        cap = None
        print("Camera released.")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
