import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
JOINTS = [12, 14, 16]  # right shoulder, elbow, wrist

sequence = []

cap = cv2.VideoCapture(0)
frame_count = 0
max_frames = 300  # ~3–5 seconds of data at 30 FPS

print("Recording... Press 'q' to stop early.")

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        keypoints = []
        for idx in JOINTS:
            lm = results.pose_landmarks.landmark[idx]
            keypoints.append([lm.x, lm.y])
        sequence.append(keypoints)
        frame_count += 1

    cv2.putText(frame, f"Recording frame {frame_count}/{max_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Recording 2D Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

sequence_np = np.array(sequence).reshape(-1, 3, 2)  # (T, 3, 2)
print("Sequence shape:", sequence_np.shape)

# Save in VideoPose3D format
data = {
    'positions_2d': {
        'S1': {
            'Walking': sequence_np
        }
    }
}
np.savez_compressed('pose2d_mediapipe.npz', **data)
print("Saved to pose2d_mediapipe.npz")
