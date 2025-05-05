import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# mediapipe landmark indicides
JOINTS = {
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        keypoints = []
        for name in JOINTS:
            landmark = results.pose_landmarks.landmark[JOINTS[name]]
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            keypoints.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        # print the keypoints for VideoPose3D
        print("2D Keypoints:", keypoints)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
