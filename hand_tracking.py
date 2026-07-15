import cv2
import mediapipe as mp
import pydirectinput
import math

# -----------------------------
# Webcam Setup
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

current_keys = set()

# -----------------------------
# Smoothing
# -----------------------------
SMOOTHING_ALPHA = 0.25
smooth_distance = None

# -----------------------------
# Thresholds
# -----------------------------
HEIGHT_DEAD_ZONE = 100

ACCEL_DISTANCE = 350
BRAKE_DISTANCE = 220

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    h, w, _ = frame.shape

    left_hand = None
    right_hand = None

    new_keys = set()

    # -----------------------------
    # Detect Hands
    # -----------------------------
    if results.multi_hand_landmarks and results.multi_handedness:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            label = results.multi_handedness[idx].classification[0].label

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Mirror correction
            if label == "Left":
                right_hand = hand_landmarks

            elif label == "Right":
                left_hand = hand_landmarks

    # -----------------------------
    # Steering Logic
    # -----------------------------
    if left_hand and right_hand:

        left_x = int(left_hand.landmark[0].x * w)
        left_y = int(left_hand.landmark[0].y * h)

        right_x = int(right_hand.landmark[0].x * w)
        right_y = int(right_hand.landmark[0].y * h)

        # Draw virtual steering wheel
        cv2.line(
            frame,
            (left_x, left_y),
            (right_x, right_y),
            (0, 255, 0),
            5
        )

        # ---------------------------------
        # Steering using wrist height
        # ---------------------------------
        height_difference = left_y - right_y

        steering_text = "STRAIGHT"

        # FIXED LEFT/RIGHT LOGIC
        if height_difference > HEIGHT_DEAD_ZONE:
            steering_text = "RIGHT"
            new_keys.add("d")

        elif height_difference < -HEIGHT_DEAD_ZONE:
            steering_text = "LEFT"
            new_keys.add("a")

        # ---------------------------------
        # Speed Control
        # ---------------------------------
        raw_distance = math.sqrt(
            (right_x - left_x) ** 2 +
            (right_y - left_y) ** 2
        )

        if smooth_distance is None:
            smooth_distance = raw_distance
        else:
            smooth_distance = (
                SMOOTHING_ALPHA * raw_distance +
                (1 - SMOOTHING_ALPHA) * smooth_distance
            )

        distance = smooth_distance

        speed_text = "CRUISE"

        if distance > ACCEL_DISTANCE:
            speed_text = "ACCELERATE"
            new_keys.add("w")

        elif distance < BRAKE_DISTANCE:
            speed_text = "BRAKE"
            new_keys.add("s")

        # ---------------------------------
        # Display Information
        # ---------------------------------
        cv2.putText(
            frame,
            f"Steering: {steering_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Height Difference: {height_difference}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {int(distance)}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Speed: {speed_text}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    else:
        cv2.putText(
            frame,
            "SHOW BOTH HANDS",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2
    )

    # ---------------------------------
    # Keyboard Handling
    # ---------------------------------
    for key in current_keys - new_keys:
        pydirectinput.keyUp(key)

    for key in new_keys - current_keys:
        pydirectinput.keyDown(key)

    current_keys = new_keys.copy()

    cv2.imshow(
        "Virtual Steering Wheel",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
for key in current_keys:
    pydirectinput.keyUp(key)

cap.release()
cv2.destroyAllWindows()