import socket
import struct
import numpy as np
import cv2
import mediapipe as mp
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 9999))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
JOINTS = [12, 14, 16]  # Right shoulder, elbow, wrist

def recv_exact(n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data += packet
    return data

def get_depth(depth_map, x, y):
    if 0 <= x < 640 and 0 <= y < 480:
        return depth_map[y, x] / 1000.0  # mm → meters
    return 0.0

last_print_time = time.time()

while True:
    try:
        jpeg_len = struct.unpack('<I', recv_exact(4))[0]
        jpeg_data = recv_exact(jpeg_len)

        depth_len = struct.unpack('<I', recv_exact(4))[0]
        depth_data = recv_exact(depth_len)
        depth_map = np.frombuffer(depth_data, dtype=np.uint16).reshape((480, 640))

        # rgb
        rgb = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # greyscale depth images
        depth_display = np.clip((depth_map / 4500.0) * 255, 0, 255).astype(np.uint8)
        depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

        if results.pose_landmarks:
            h, w, _ = frame_rgb.shape
            joints_xyz = []

            for idx in JOINTS:
                lm = results.pose_landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)

                z = get_depth(depth_map, x, y)
                joints_xyz.append((lm.x, lm.y, z))

                # green dots on the joints
                cv2.circle(depth_display, (x, y), 6, (0, 255, 0), -1)

            # lines between joints
            for i in range(2):
                pt1 = (int(results.pose_landmarks.landmark[JOINTS[i]].x * w),
                       int(results.pose_landmarks.landmark[JOINTS[i]].y * h))
                pt2 = (int(results.pose_landmarks.landmark[JOINTS[i + 1]].x * w),
                       int(results.pose_landmarks.landmark[JOINTS[i + 1]].y * h))
                cv2.line(depth_display, pt1, pt2, (0, 255, 0), 2)

            # print the coords
            if time.time() - last_print_time > 3.0:
                print("3D Joint Coordinates (shoulder → wrist):")
                for name, joint in zip(["Shoulder", "Elbow", "Wrist"], joints_xyz):
                    print(f"  {name}: x={joint[0]:.3f}, y={joint[1]:.3f}, z={joint[2]:.3f} m")
                print("-" * 40)
                last_print_time = time.time()

        # live depth feed with overlays
        cv2.imshow("Kinect Depth + Joints", depth_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)
        break

sock.close()
cv2.destroyAllWindows()
