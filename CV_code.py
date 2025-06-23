import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import threading

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------ Using Ngrok to make HTTP connection ---------------------------------------------
last_sent_gesture_id = None
gesture_lock = threading.Lock()

# -------------------- This NGROK_URL should be changed to match the one send from the PC2 -----------------------------
NGROK_URL = ("https://66da-46-122-98-75.ngrok-free.app/gesture")

def send_gesture_to_server(gesture_id):
    try:
        print(f"Sending to: {NGROK_URL} — Gesture ID: {gesture_id}")
        response = requests.post(NGROK_URL, json={"gesture": gesture_id})

        if response.status_code == 200:
            print(f"Sent Gesture ID: {gesture_id}")
        else:
            print(f"Server Error [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"Failed to send gesture: {e}")

# ----------------------------------------------------------------------------------------------------------------------

# initialize video capture and set properties
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.8, min_tracking_confidence=0.8)
# (if too much false detection of landmarks occur make detection higher)

wrist_movement_limit = 3 # angle change for wrist rotation
init_wrist_distance = None
prev_wrist_distance = None
max_left_percent = 60  # max rotation to left (to avoid landmarks overlapping)

# gesture labels
GESTURE_LABELS = {
    0: 'Unknown',
    1: 'Power',
    2: 'Trust',
    3: 'Attention',
    4: 'Promise',
    5: 'Okay',
    6: 'Victory',
    7: 'ILoveYou',
    8: 'Hello'
}

# function to recognize gesture based on hand landmarks
def recognize_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # check if fingers are open
    thumb_is_open = thumb_ip.y > thumb_tip.y
    index_is_open = index_tip.y < landmarks[6].y
    middle_is_open = middle_tip.y < landmarks[10].y
    ring_is_open = ring_tip.y < landmarks[14].y
    pinky_is_open = pinky_tip.y < landmarks[18].y

    fingers_status = [thumb_is_open, index_is_open, middle_is_open, ring_is_open, pinky_is_open]

    # gesture, position of fingers
    if all(not status for status in fingers_status):
        return 1  # Power
    elif all(status for status in fingers_status):
        return 2  # Trust
    elif index_is_open and not any([middle_is_open, ring_is_open, pinky_is_open]):
        return 3  # Attention
    elif pinky_is_open and not any([index_is_open, middle_is_open, ring_is_open]):
        return 4  # Promise
    elif (thumb_is_open and not any([index_is_open, middle_is_open, ring_is_open, pinky_is_open])
          and thumb_tip.y < thumb_ip.y + 0.1):
        return 5  # Okay
    elif index_is_open and middle_is_open and not any([ring_is_open, pinky_is_open]):
        return 6  # Victory
    elif thumb_is_open and index_is_open and pinky_is_open and not any([middle_is_open, ring_is_open]):
        return 7  # ILoveYou
    else:
        return 0  # Unknown

# limit gesture recognition to 2 seconds (to avoid spam)
previous_gesture_id = None
current_gesture_id = None
gesture_start_time = None
GESTURE_DISPLAY_DURATION = 2

# countdown (to arrange the hand to open palm position)
countdown_start_time = time.time()
countdown_duration = 10
pre_countdown_duration = 3

gesture_cooldown = 3
last_gesture_time = 0

hello_counter = 0
hello_required_count = 2
prev_rotation_direction = None

while True:
    success, frame = cap.read()
    if success:
        # get elapsed time since countdown started
        elapsed_time = time.time() - countdown_start_time
        remaining_time = max(0, int(countdown_duration - elapsed_time))

        if elapsed_time < pre_countdown_duration:
            text = "Starting countdown..."
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 255), 3, cv2.LINE_AA)

        elif elapsed_time >= pre_countdown_duration and elapsed_time < (pre_countdown_duration + countdown_duration):
            countdown_elapsed = elapsed_time - pre_countdown_duration
            remaining_time = max(0, int(countdown_duration - countdown_elapsed))
            if remaining_time > 0:
                text = str(remaining_time)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                            (0, 0, 255), 5, cv2.LINE_AA)

        elif elapsed_time >= (pre_countdown_duration + countdown_duration):
            # convert frame form BGR (OpenCV) to RGB (for MediaPipe)
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # process frame to detect hands
            result = hand.process(RGB_frame)

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    # check if detected hand is right hand (because of camera it needs to be inverted)
                    if handedness.classification[0].label == 'Left':
                        recognized_gesture_id = recognize_gesture(hand_landmarks)

                        # ------------------------------------------------------------------------------------------
                        # -------------------------- Recognize gestures --------------------------------------------
                        if recognized_gesture_id == current_gesture_id:
                            if time.time() - gesture_start_time >= GESTURE_DISPLAY_DURATION:
                                if previous_gesture_id != current_gesture_id:
                                    previous_gesture_id = current_gesture_id
                                    print(f"Gesture: {current_gesture_id}")
                                    threading.Thread(target=send_gesture_to_server,
                                                     args=(current_gesture_id,), daemon=True).start()
                        else:
                            current_gesture_id = recognized_gesture_id
                            gesture_start_time = time.time()

                        gesture_label = GESTURE_LABELS.get(previous_gesture_id, 'Unknown')
                        # --------------------------------------------------------------------------
                        # draw landmarks and connections
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Draw the gesture label in the top left corner
                        cv2.putText(frame, gesture_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

                        # draw bounding box and landmark numbers
                        landmarks_positions = []
                        image_height, image_width, _ = frame.shape

                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            x = int(landmark.x * image_width)
                            y = int(landmark.y * image_height)
                            landmarks_positions.append((x, y))

                        x_min = min([p[0] for p in landmarks_positions])
                        y_min = min([p[1] for p in landmarks_positions])
                        x_max = max([p[0] for p in landmarks_positions])
                        y_max = max([p[1] for p in landmarks_positions])

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            landmark_x = int(landmark.x * image_width)
                            landmark_y = int(landmark.y * image_height)
                            cv2.putText(frame, str(idx), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        # calculate the box width
                        box_width = x_max - x_min

                        # ------------------------------------------------------------------------------------------
                        # -------------------------- Recognize wrist movement --------------------------------------
                        # calculate wrist position based on detected hand landmarks (0,5,17) palm
                        wrist_landmarks_points = [0, 5, 17]
                        wrist_landmarks = [(hand_landmarks.landmark[i].x * image_width,
                                            hand_landmarks.landmark[i].y * image_height)
                                           for i in wrist_landmarks_points]

                        # extract coordinates of the landmarks
                        (x0, y0), (x5, y5), (x17, y17) = wrist_landmarks

                        # calculate distance between landmarks 5 and 17
                        distance_5_17 = np.sqrt((x17 - x5) ** 2 + (y17 - y5) ** 2)

                        if init_wrist_distance is None:
                            init_wrist_distance = distance_5_17

                        # calculate percentage of rotation based on initial distance
                        rotation_percentage = (distance_5_17 / box_width) * 100

                        if rotation_percentage <= max_left_percent:
                            percentage_change = 0
                            if prev_wrist_distance is not None:
                                # calculate the change in wrist distance (movement threshold)
                                percentage_change = abs(rotation_percentage - (prev_wrist_distance / box_width) * 100)

                            if percentage_change >= wrist_movement_limit:
                                if distance_5_17 < prev_wrist_distance:
                                    rotation_direction = "left"
                                else:
                                    rotation_direction = "right"

                                # check if direction changed AND cooldown passed
                                current_time = time.time()
                                if prev_rotation_direction is not None and \
                                        rotation_direction != prev_rotation_direction and \
                                        current_time - last_gesture_time >= gesture_cooldown:
                                    hello_counter += 1

                                    if hello_counter >= hello_required_count:
                                        current_gesture_id = 8
                                        if previous_gesture_id != current_gesture_id:
                                            previous_gesture_id = current_gesture_id
                                            last_hello_time = time.time()
                                            threading.Thread(target=send_gesture_to_server,
                                                             args=(2,), daemon=True).start()
                                            print(f"Gesture: {current_gesture_id}")
                                            threading.Thread(target=send_gesture_to_server,
                                                             args=(current_gesture_id,), daemon=True).start()

                                        hello_counter = 0  # reset after successful Hello
                                    last_gesture_time = current_time  # update time

                                # update previous direction
                                prev_rotation_direction = rotation_direction

                            # update previous distance regardless of condition
                            prev_wrist_distance = distance_5_17
                        # ------------------------------------------------------------------------------------------

        # display output frame
        cv2.imshow("Hand Gesture Recognition", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

# close video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()