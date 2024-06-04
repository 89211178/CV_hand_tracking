import cv2
import mediapipe as mp
import numpy as np

# initialize video capture and set properties
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

# get angle of fingers
def unit_vector(vector):
    return vector / np.linalg.norm(vector)
def calculate_angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# define finger landmark points
finger_points = {
    'thumb': [0, 1, 2, 3, 4],
    'index': [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring': [0, 13, 14, 15, 16],
    'pinky': [0, 17, 18, 19, 20]
}

movement_limit = 5 # angle change to limit spam (because of twitching)
previous_finger_angles = {}
for finger, indices in finger_points.items():
    previous_finger_angles[finger] = [None] * (len(indices) - 2)

# limit how close hand must be (to avoid twitching)
limit_height = 200
limit_width = 200

# track if message has been printed previously to avoid spam
hand_close_printed = False

while True:
    success, frame = cap.read()
    if success:
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
                        landmarks_positions.append((x, y))

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

                    # check if bounding box is large enough (to avoid twitching)
                    if box_width > limit_width and box_height > limit_height:
                        # calculate x, y and angles for fingers
                        for finger, indices in finger_points.items():
                            for i in range(len(indices) - 2):
                                v1 = np.array(landmarks_positions[indices[i + 1]]) - np.array(landmarks_positions[indices[i]])
                                v2 = np.array(landmarks_positions[indices[i + 2]]) - np.array(landmarks_positions[indices[i + 1]])
                                angle = calculate_angle(v1, v2)

                                if previous_finger_angles[finger][i] is None or abs(angle - previous_finger_angles[finger][i]) >= movement_limit:
                                    for idx, landmark in enumerate(hand_landmarks.landmark):
                                        x_relative = landmarks_positions[indices[i + 1]][0] - x_min_init
                                        y_relative = landmarks_positions[indices[i + 1]][1] - y_min_init
                                        print(f"landmark: {idx}): x={x_relative}, y={y_relative}, angle= {angle:.2f} degrees")
                                previous_finger_angles[finger][i] = angle

                        # draw bounding box
                        cv2.rectangle(frame, (x_min_init, y_min_init), (x_max_init, y_max_init), (0, 255, 0), 2)
                        # draw landmarks and connections
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        if not hand_close_printed:
                            print("Hand is too far away")
                            hand_close_printed = True

        # display output frame with landmarks and bounding box
        cv2.imshow("Right Hand Detection", frame)

        # break if 'q' key is pressed on the camera window
        if cv2.waitKey(1) == ord('q'):
            break

# close video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()