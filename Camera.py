import cv2
import mediapipe as mp

# initialize video capture and set properties
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
                        print(f"Landmark {idx}: x={x}, y={y}")

                        # calculate bounding box to be resized based on movement
                    x_min = min([p[0] for p in landmarks_positions])
                    y_min = min([p[1] for p in landmarks_positions])
                    x_max = max([p[0] for p in landmarks_positions])
                    y_max = max([p[1] for p in landmarks_positions])

                    # draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    # draw landmarks and connections
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    break

                    # display output frame with landmarks and bounding box
        cv2.imshow("Right Hand Detection", frame)

        # break if 'q' key is pressed on the camera window
        if cv2.waitKey(1) == ord('q'):
            break

        # close video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()