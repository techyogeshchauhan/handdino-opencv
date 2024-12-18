# # import numpy as np
# # import cv2

# # import math
# # import pyautogui


# # # Open Camera
# # capture = cv2.VideoCapture(0)

# # while capture.isOpened():

# #     # Capture frames from the camera
# #     ret, frame = capture.read()

# #     # Get hand data from the rectangle sub window
# #     cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
# #     crop_image = frame[100:300, 100:300]

# #     # Apply Gaussian blur
# #     blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

# #     # Change color-space from BGR -> HSV
# #     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# #     # Create a binary image with where white will be skin colors and rest is black
# #     mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

# #     # Kernel for morphological transformation
# #     kernel = np.ones((5, 5))

# #     # Apply morphological transformations to filter out the background noise
# #     dilation = cv2.dilate(mask2, kernel, iterations=1)
# #     erosion = cv2.erode(dilation, kernel, iterations=1)

# #     # Apply Gaussian Blur and Threshold
# #     filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
# #     ret, thresh = cv2.threshold(filtered, 127, 255, 0)
# # #####
# #     # Find contours
# #     #image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #     contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# #     try:
# #         # Find contour with maximum area
# #         contour = max(contours, key=lambda x: cv2.contourArea(x))

# #         # Create bounding rectangle around the contour
# #         x, y, w, h = cv2.boundingRect(contour)
# #         cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

# #         # Find convex hull
# #         hull = cv2.convexHull(contour)

# #         # Draw contour
# #         drawing = np.zeros(crop_image.shape, np.uint8)
# #         cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
# #         cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

# #         # Fi convexity defects
# #         hull = cv2.convexHull(contour, returnPoints=False)
# #         defects = cv2.convexityDefects(contour, hull)

# #         # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
# #         # tips) for all defects
# #         count_defects = 0

# #         for i in range(defects.shape[0]):
# #             s, e, f, d = defects[i, 0]
# #             start = tuple(contour[s][0])
# #             end = tuple(contour[e][0])
# #             far = tuple(contour[f][0])

# #             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
# #             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
# #             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
# #             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

# #             # if angle >= 90 draw a circle at the far point
# #             if angle <= 90:
# #                 count_defects += 1
# #                 cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

# #             cv2.line(crop_image, start, end, [0, 255, 0], 2)

# #         # Press SPACE if condition is match

# #         if count_defects >= 4:
# #                 pyautogui.press('space')
# #                 cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

# #         #PLAY RACING GAMES (WASD)
# #         """
# #         if count_defects == 1:
# #             pyautogui.press('w')
# #             cv2.putText(frame, "W", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
# #         if count_defects == 2:
# #             pyautogui.press('s')
# #             cv2.putText(frame, "S", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
# #         if count_defects == 3:
# #             pyautogui.press('aw')
# #             cv2.putText(frame, "aw", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
# #         if count_defects == 4:
# #             pyautogui.press('dw')
# #             cv2.putText(frame, "dw", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
# #         if count_defects == 5:
# #             pyautogui.press('s')
# #             cv2.putText(frame, "s", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
# #         """

# #     except:
# #         pass

# #     # Show required images
# #     cv2.imshow("Gesture", frame)

# #     # Close the camera if 'q' is pressed
# #     if cv2.waitKey(1) == ord('q'):
# #         break

# # capture.release()
# # cv2.destroyAllWindows()



# import cv2
# import math
# import pyautogui
# import mediapipe as mp

# # Initialize MediaPipe Hand module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_draw = mp.solutions.drawing_utils

# # Open Camera
# capture = cv2.VideoCapture(0)

# while capture.isOpened():
#     ret, frame = capture.read()
#     if not ret:
#         continue

#     # Flip the frame for better visualization
#     frame = cv2.flip(frame, 1)

#     # Convert the frame to RGB (MediaPipe uses RGB format)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame and get hand landmarks
#     results = hands.process(rgb_frame)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw hand landmarks
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Find the landmarks we need (e.g., tip of the index and middle fingers)
#             landmarks = hand_landmarks.landmark
#             index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

#             # Calculate distance between index and middle finger tips
#             dx = index_tip.x - middle_tip.x
#             dy = index_tip.y - middle_tip.y
#             distance = math.sqrt(dx ** 2 + dy ** 2)

#             # Gesture detection based on distance (e.g., if fingers are close together)
#             if distance < 0.05:  # Adjust threshold as needed
#                 pyautogui.press('space')
#                 cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

#     # Display the frame with hand landmarks
#     cv2.imshow("Hand Gesture", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()




import cv2
import math
import pyautogui
import mediapipe as mp
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open Camera
capture = cv2.VideoCapture(0)
capture.set(3, 640)  # Width
capture.set(4, 480)  # Height

prev_time = time.time()

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        continue

    # Flip the frame for better visualization
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Find the landmarks we need (e.g., tip of the index and middle fingers)
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate distance between index and middle finger tips
            dx = index_tip.x - middle_tip.x
            dy = index_tip.y - middle_tip.y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # Gesture detection based on distance
            if distance < 0.05:  # Adjust threshold as needed
                pyautogui.press('space')
                cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Measure frame rate
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with hand landmarks
    cv2.imshow("Hand Gesture", frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
