import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import time

# === Pose lifting model (for shoulder, elbow, wrist only) ===
class PoseLifter(nn.Module):
    def __init__(self, input_dim=6, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Load pose lifter model ===
model = PoseLifter()
try:
    model.load_state_dict(torch.load('pose_lifter_weights.pth', map_location='cpu'))
    print("Loaded pretrained weights.")
except FileNotFoundError:
    print("Using random weights! 3D output will be nonsense.")
model.eval()

# === MediaPipe Pose setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
JOINTS = [12, 14, 16]  # right shoulder, elbow, wrist

# === Kinect depth intrinsics (for Xbox 360 Kinect) ===
def hand_center_3d(depth_map, box, fx=594.21, fy=591.04, cx=339.5, cy=242.7):
    x1, y1, x2, y2 = box
    depth_crop = depth_map[y1:y2, x1:x2]
    valid = depth_crop[depth_crop > 0]

    if valid.size == 0:
        return None

    z = np.mean(valid) / 1000.0  # mm → meters
    cx_px = (x1 + x2) // 2
    cy_px = (y1 + y2) // 2

    x = (cx_px - cx) * z / fx
    y = (cy_px - cy) * z / fy
    return np.array([x, y, z])

# === Webcam + depth stream setup ===
cap = cv2.VideoCapture(0)  # RGB camera
# TODO: replace this with actual depth map input
depth_map = np.zeros((480, 640), dtype=np.uint16)  # Dummy placeholder

# 2d plot
fig = plt.figure(figsize=(10, 5))
ax2d = fig.add_subplot(121)
joint2d_line, = ax2d.plot([], [], 'ro-', linewidth=2, label='2D Arm (MP)')
ax2d.set_xlim(0, 1)
ax2d.set_ylim(1, 0)
ax2d.set_title("2D Right Arm")
ax2d.set_xlabel("X (normalized)")
ax2d.set_ylabel("Y (normalized)")
ax2d.legend()

# 3d plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([], [], [], marker='o', label=label)[0] for label in ['Shoulder', 'Elbow', 'Wrist']]
skeleton_line, = ax.plot([], [], [], color='black', linewidth=2, label='Arm Skeleton')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Live 3D Right Arm Pose')
ax.legend()

history = [deque(maxlen=5) for _ in range(3)]
last_print_time = time.time()

print("Running real-time 2D + depth-based 3D hand tracking. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        keypoints_2d = []
        for idx in JOINTS:
            lm = results.pose_landmarks.landmark[idx]
            x = lm.x
            y = lm.y
            keypoints_2d.extend([x, y])
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)

        # bounding box for the hand
        elbow_landmark = results.pose_landmarks.landmark[14]
        wrist_landmark = results.pose_landmarks.landmark[16]
        ex, ey = int(elbow_landmark.x * w), int(elbow_landmark.y * h)
        wx, wy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
        dx, dy = wx - ex, wy - ey

        scale = 0.45
        hx = int(wx + dx * scale)
        hy = int(wy + dy * scale)
        box_size = 120
        x1 = max(0, hx - box_size // 2)
        y1 = max(0, hy - box_size // 2)
        x2 = min(w, hx + box_size // 2)
        y2 = min(h, hy + box_size // 2)
        hand_box = (x1, y1, x2, y2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Hand", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # === 3D prediction of shoulder, elbow, wrist ===
        if len(keypoints_2d) == 6:
            input_tensor = torch.tensor([keypoints_2d], dtype=torch.float32)
            with torch.no_grad():
                pred_3d = model(input_tensor).view(3, 3)

            for i, joint in enumerate(pred_3d):
                history[i].append(joint.numpy())

            joint2d = np.array(keypoints_2d).reshape(3, 2)
            joint2d_line.set_data(joint2d[:, 0], joint2d[:, 1])

            for i, line in enumerate(lines):
                data = np.array(history[i])
                if len(data) > 0:
                    line.set_data(data[:, 0], data[:, 1])
                    line.set_3d_properties(data[:, 2])

            if all(len(h) > 0 for h in history):
                joints_xyz = [h[-1] for h in history]
                joints_xyz = np.stack(joints_xyz)
                skeleton_line.set_data(joints_xyz[:, 0], joints_xyz[:, 1])
                skeleton_line.set_3d_properties(joints_xyz[:, 2])

                # === Get hand 3D coordinates from depth ===
                hand_pos = hand_center_3d(depth_map, hand_box)
                if hand_pos is not None and time.time() - last_print_time > 1.5:
                    print("3D Arm Coordinates:")
                    for label, pt in zip(["Shoulder", "Elbow", "Wrist"], joints_xyz):
                        print(f"  {label}: x={pt[0]:+.3f}, y={pt[1]:+.3f}, z={pt[2]:+.3f}")
                    print(f"  Hand:    x={hand_pos[0]:+.3f}, y={hand_pos[1]:+.3f}, z={hand_pos[2]:+.3f}")
                    print("-" * 40)
                    last_print_time = time.time()

            plt.draw()
            plt.pause(0.001)

    cv2.imshow("Webcam - 2D Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
