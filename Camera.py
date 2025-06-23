import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# initialize video capture and set properties
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.8, min_tracking_confidence=0.8)
# (if too much false detection of landmarks occur make detection higher)

# to store landmark positions for smoothing (to avoid even more false detection)
landmark_smoothing = {i: deque(maxlen=5) for i in range(21)}

wrist_movement_limit = 3 # angle change for wrist rotation
init_wrist_distance = None
prev_wrist_distance = None
max_left_percent = 60  # max rotation to left (to avoid landmarks overlapping)

# define finger landmark points
finger_points = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

# define distance (form 0 tp [4,8,12,16,20]) for determining finger status
# to filter data to only specific options
percentage_positions = {
    'thumb': {'closed': (50, 59), 'half_closed': (60, 69),
              'half': (70, 79), 'half_open': (80, 89),
              'open': (90, 100)},
    'index': {'closed': (50, 59), 'half_closed': (60, 69),
              'half': (70, 79), 'half_open': (80, 89),
              'open': (90, 100)},
    'middle': {'closed': (50, 59), 'half_closed': (60, 69),
              'half': (70, 79), 'half_open': (80, 89),
              'open': (90, 100)},
    'ring': {'closed': (50, 59), 'half_closed': (60, 69),
              'half': (70, 79), 'half_open': (80, 89),
              'open': (90, 100)},
    'pinky': {'closed': (50, 59), 'half_closed': (60, 69),
              'half': (70, 79), 'half_open': (80, 89),
              'open': (90, 100)},
}

# store information for comparison
previous_status = {finger: None for finger in finger_points.keys()}
previous_distance = {finger: None for finger in finger_points.keys()}
initial_finger_distances = {finger: None for finger in finger_points.keys()}

# countdown (to arrange the fingers to open position)
countdown_start_time = time.time()
countdown_duration = 10

while True:
    success, frame = cap.read()
    if success:
        # get elapsed time
        elapsed_time = time.time() - countdown_start_time
        remaining_time = max(0, int(countdown_duration - elapsed_time))

        if remaining_time > 0: # show countdown
            # display countdown in middle of screen
            text = str(remaining_time)
            # position on the screen
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255), 5, cv2.LINE_AA)

        else: # starts finger detection after countdown
            # convert frame form BGR (OpenCV) to RGB (for MediaPipe)
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # process frame to detect hands
            result = hand.process(RGB_frame)

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    # check if detected hand is right hand (because of camera it needs to be inverted)
                    if handedness.classification[0].label == 'Left':
                        # collect all landmarks
                        landmarks_positions = []
                        image_height, image_width, _ = frame.shape

                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            x = int(landmark.x * image_width)
                            y = int(landmark.y * image_height)
                            landmark_smoothing[idx].append((x, y))
                            # to smoothen landmarks before appending
                            smoothed_position = np.mean(landmark_smoothing[idx], axis=0)
                            landmarks_positions.append(tuple(smoothed_position.astype(int)))

                        # calculate bounding box to be resized based on movement
                        x_min = min([p[0] for p in landmarks_positions])
                        y_min = min([p[1] for p in landmarks_positions])
                        x_max = max([p[0] for p in landmarks_positions])
                        y_max = max([p[1] for p in landmarks_positions])
                        # bounding box based on the current hand position
                        x_min_init, y_min_init, x_max_init, y_max_init = x_min, y_min, x_max, y_max

                        # calculate width and height of bounding box
                        box_width = x_max_init - x_min_init
                        box_height = y_max_init - y_min_init

                        # ---------------------------------------------------------------------------------------------------
                        wrist_position = landmarks_positions[0]
                        # store initial finger distances if not already stored
                        if any(distance is None for distance in initial_finger_distances.values()):
                            for finger, indices in finger_points.items():
                                if indices[-1] in [4, 8, 12, 16, 20]:
                                    finger_position = landmarks_positions[indices[-1]]
                                    # normalize position with respect to bounding box
                                    # to avoid change of distance based on how close hand is to camera
                                    finger_position = ((finger_position[0] - x_min_init) / box_width,
                                                       (finger_position[1] - y_min_init) / box_height)
                                    wrist_position = ((wrist_position[0] - x_min_init) / box_width,
                                                      (wrist_position[1] - y_min_init) / box_height)
                                    distance_1 = np.linalg.norm(np.array(finger_position) - np.array(wrist_position))
                                    initial_finger_distances[finger] = distance_1
                                    #print(f"Finger: {finger}, Distance={distance_1:.2f}") #(for testing)

                        # calculate distances for landmarks [4,8,12,16,20] to landmark 0 (wrist_position)
                        for finger, indices in finger_points.items():
                            if indices[-1] in [4, 8, 12, 16, 20]:
                                # get initial distance for this finger
                                distance_1 = initial_finger_distances[finger]
                                finger_position = landmarks_positions[indices[-1]]
                                # normalize position with respect to bounding box
                                # to avoid change of distance based on how close hand is to camera
                                normalized_finger_position = ((finger_position[0] - x_min_init) / box_width,
                                                              (finger_position[1] - y_min_init) / box_height)
                                normalized_wrist_position = ((wrist_position[0] - x_min_init) / box_width,
                                                             (wrist_position[1] - y_min_init) / box_height)

                                # calculate Euclidean distance
                                distance = np.linalg.norm(
                                    np.array(normalized_finger_position) - np.array(normalized_wrist_position))

                                #print(f"Distance={distance:.2f} , Distance_1={distance_1:.2f}") #(for testing)

                                # calculate percentage change of distance from initial distance for each finger
                                percentage_change_distance = (distance / distance_1) * 100

                                # get status of finger based on percentage change
                                finger_status = None
                                for status, (percent_range_start, percent_range_end) in percentage_positions[
                                    finger].items():
                                    if percent_range_start <= percentage_change_distance <= percent_range_end:
                                        finger_status = status
                                        break

                                # update and print finger status if it has changed
                                if finger_status and (finger_status != previous_status[finger] or
                                                      previous_distance[finger] is None or
                                                      abs(distance - previous_distance[finger]) > 0.06):
                                    if finger_status != previous_status[finger] or previous_distance[finger] is None:
                                        print(f"Finger: {finger}, Distance={distance:.2f}, Percentage={percentage_change_distance:.2f}%, Status: {finger_status}")
                                        previous_status[finger] = finger_status
                                        previous_distance[finger] = distance
                                    elif abs(distance - previous_distance[finger]) > 0.06:
                                        previous_distance[finger] = distance
                        # --------------------------------------------------------------------------------------------------
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
                                percentage_change = abs(rotation_percentage - (prev_wrist_distance / box_width) * 100)

                            if percentage_change >= wrist_movement_limit:
                                if distance_5_17 < prev_wrist_distance:
                                    rotation_direction = "left"
                                else:
                                    rotation_direction = "right"
                                print(f"Wrist rotation: {rotation_percentage:.2f}% ({rotation_direction})")
                            prev_wrist_distance = distance_5_17
                        # --------------------------------------------------------------------------------------------------

                        # draw bounding box
                        cv2.rectangle(frame, (x_min_init, y_min_init), (x_max_init, y_max_init), (0, 255, 0), 2)
                        # draw landmarks and connections
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # draw landmark numbers (for more accurate testing)
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            landmark_x = int(landmark.x * image_width)
                            landmark_y = int(landmark.y * image_height)

                            cv2.putText(frame, str(idx), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # display output frame with landmarks and bounding box
        cv2.imshow("Right Hand Detection", frame)

        # break if 'q' key is pressed on the camera window
        if cv2.waitKey(1) == ord('q'):
            break

# close video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()