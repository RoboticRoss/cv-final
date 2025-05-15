import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import time
from datetime import datetime
import tensorflow as tf
import keras
from PIL import Image
from models import YourModel, VGGModel
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver
import hyperparameters as hp
from skimage.io import imread
from skimage.transform import resize
from lime import lime_image
from skimage.segmentation import mark_boundaries
import freenect
import serial

clawModel = VGGModel()
clawModel(tf.keras.Input(shape=(224, 224, 3)))
clawModel.vgg16.load_weights('/Users/rossgoldbaum/Desktop/CS1430_Projects/cv-final/vgg16_imagenet.weights.h5')
clawModel.head.load_weights('/Users/rossgoldbaum/Desktop/CS1430_Projects/cv-final/vgg.e018-acc0.9679.weights.h5')
clawModel.compile(
        optimizer=clawModel.optimizer,
        loss=clawModel.loss_fn,
        metrics=["sparse_categorical_accuracy"])
clawAngle = 55
closeLabel = "closed"

# !!!! THIS CHANGES EVERY TIME THE DAMN ARDUINO GETS PLUGGED IN
ser = serial.Serial('/dev/tty.usbmodem1201', 9600, timeout=1)

time.sleep(2)

#Classification for open/closed
def classify(depth_norm, model):
    if len(depth_norm.shape) == 2:
        depth_norm = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(depth_norm, (224, 224)).astype(np.float32)
    img = keras.applications.vgg16.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)
    prediction = np.argmax(prediction, axis=1)

    if prediction:
        return 'open'
    else:
        return 'closed'



# Kinect utilities
def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.uint16)

def get_kinect_rgb():
    rgb_frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

# pose lifting -> tried to maybe estimate the 3d from 2d images, but didn't really work
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

model = PoseLifter()
try:
    model.load_state_dict(torch.load('pose_lifter_weights.pth', map_location='cpu'))
    print("Loaded pretrained weights.")
except FileNotFoundError:
    print("Using random weights! 3D output will be nonsense.")
model.eval()

# mediapipe pose stuff
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# joint indices (right shoulder, elbow, wrist)
JOINTS = [12, 14, 16]  # right shoulder, elbow, wrist

# Start webcam
cap = cv2.VideoCapture(0)

# 2D subplot
fig = plt.figure(figsize=(10, 5))
ax2d = fig.add_subplot(121)
joint2d_line, = ax2d.plot([], [], 'ro-', linewidth=2, label='2D Arm (MP)')
ax2d.set_xlim(0, 1)
ax2d.set_ylim(1, 0)  # inverted Y-axis
ax2d.set_title("2D Right Arm")
ax2d.set_xlabel("X (normalized)")
ax2d.set_ylabel("Y (normalized)")
ax2d.legend()

# 3D subplot
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# lines = [ax.plot([], [], [], marker='o', label=label)[0] for label in ['Shoulder', 'Elbow', 'Wrist']]
# skeleton_line, = ax.plot([], [], [], color='black', linewidth=2, label='Arm Skeleton')
# ax.set_xlim([-0.5, 0.5])
# ax.set_ylim([-0.5, 0.5])
# ax.set_zlim([-0.5, 0.5])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Live 3D Right Arm Pose')
# ax.legend()

# Servo limits
BASE_RANGE = (5, 155)
SHOULDER_RANGE = (10, 90)  # horizontal to vertical
ELBOW_RANGE = (160, 160)   # fixed elbow for simplicity (fully extended)

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def match_training_style(depth_img):
    recolored = np.zeros_like(depth_img)
    lowest = np.min(depth_img)
    recolored[depth_img > 100] = 0
    recolored[(depth_img <= lowest+1)] = 40
    recolored[(depth_img > lowest+1) & (depth_img <= 100)] = 150
    return recolored

def compute_servo_angles(x, y, z, simple=False):
    L1 = 0.15
    L2 = 0.15
    
    if simple : #For physical debugging
        base_angle = 85 if np.sqrt(x**2 + z**2) < 0.05 else np.degrees(np.arctan2(x, z/8)) + 90
        base_angle = clamp(base_angle, *BASE_RANGE)
        y_scaled = -y * 5
        shoulder_angle = np.interp(z, [0.68, 0.9], [5, 85]) + y_scaled * 30
        shoulder_angle = clamp(shoulder_angle, *SHOULDER_RANGE)
        elbow_angle = np.interp(z, [0.68, 0.9], [160, 40]) + y_scaled * 20
        return clamp(base_angle, *BASE_RANGE), clamp(shoulder_angle, *SHOULDER_RANGE), clamp(elbow_angle, *ELBOW_RANGE)
    
    else :  
        base_angle = np.degrees(np.arctan2(x, z))
        base_angle = clamp(base_angle + 90, *BASE_RANGE)  # center around 90 degrees

        r = np.sqrt(z**2 + y**2)
        r = np.clip(r, 1e-6, L1 + L2 - 1e-6)

        #law of cosines for angle measurement
        cos_angle = (L1**2 + L2**2 - r**2) / (2 * L1 * L2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        elbow_angle_rad = np.arccos(cos_angle)
        elbow_angle = 180 - np.degrees(elbow_angle_rad)

        cos_theta = (L1**2 + r**2 - L2**2) / (2 * L1 * r)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        shoulder_offset = np.arccos(cos_theta)
        shoulder_lift = np.arctan2(y, z)
        shoulder_angle = np.degrees(shoulder_lift + shoulder_offset)

        return (
            clamp(base_angle, *BASE_RANGE),
            clamp(shoulder_angle, *SHOULDER_RANGE),
            clamp(elbow_angle, *ELBOW_RANGE)
        )

# store the joint history
history = [deque(maxlen=5) for _ in range(3)]

print("Press Q to quit.")

capture_interval = 0.35 # seconds
last_print_time = time.time()
last_capture_time = time.time()
image_counter = 0
influence = 0

while True:
    frame = get_kinect_rgb()
    if frame is None:
        continue

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    depth_vis = get_kinect_rgb()


    if results.pose_landmarks:
        keypoints_2d = []
        for idx in JOINTS:
            lm = results.pose_landmarks.landmark[idx]
            x = lm.x
            y = lm.y
            keypoints_2d.extend([x, y])
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)

        # bounding box
        # as a failsafe for the bounding box with the cnn, we used this to draw a bounding box around the hand
        # by using the wrist and elbow joints. We knwo the hand is going to be at the end of the wrist and along
        # the same line that contains the elbow and wrist.
        elbow_landmark = results.pose_landmarks.landmark[14]
        wrist_landmark = results.pose_landmarks.landmark[16]

        ex, ey = int(elbow_landmark.x * w), int(elbow_landmark.y * h)
        wx, wy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)

        dx = wx - ex
        dy = wy - ey

        scale = 0.45 
        hx = int(wx + dx * scale)
        hy = int(wy + dy * scale)-30
        
        #BBBoounding bboxxx
        box_size = 110 # change with camera res
        x1 = max(0, hx - box_size // 2)
        y1 = max(0, hy - box_size // 2)
        x2 = min(w, hx + box_size // 2)
        y2 = min(h, hy + box_size // 2)

        # draw the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Hand", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        depth_frame = get_depth() # helper method call to get the depth frame
        hand_depth = depth_frame[y1:y2, x1:x2] # only look at the bounding box
        
        depth_vis = np.clip((depth_frame / 4500.0) * 255, 0, 255).astype(np.uint8)
        # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


        # filter out zero-depth (invalid)
        valid_depth = hand_depth[hand_depth > 0]

        if valid_depth.size > 0:
            # avg_depth = np.mean(valid_depth) # average it out
            # changed to take the first top 20 intensity values instead of all the average (more acfurate)
            depth_sorted = np.sort(valid_depth.flatten())
            num_closest = min(20, len(depth_sorted))
            avg_depth = np.mean(depth_sorted[:num_closest])

            # center of the bounding box in pixel coordinates
            hand_px = (hx, hy)

            # real-world coordinates
            fx, fy = 594.21, 591.04 
            cx, cy = w // 2, h // 2  #center of image

            z = avg_depth / 1000.0
            x = (hand_px[0] - cx)/640 *0.5
            y = (hand_px[1] - cy)/480 * 0.6 
            base, shoulder, elbow = compute_servo_angles(x, y, z)
            
            # Debugging
            # print(f"Base: {base:.1f}°, Shoulder: {shoulder:.1f}°, Elbow: {elbow:.1f}°")
            # print(f"X: {x:.4f}°, Y: {y:.4f}°, Z: {z:.4f}°")
            
            #.788 z corresponds to fully extended out, 0.9 is all the way back
            # clip z from .7 to 1.
            
            # z from 0.7 to 0.91
            # x from -0.28 to 0.28
            # y from -0.35 to 0.35
            
            
            # angles = [40+100*z, 60, 100, 30]
            angles = [base, shoulder, elbow, clawAngle]
            angle_str = 'START,' + ','.join(map(str, angles)) + '\n'
            ser.write(angle_str.encode('utf-8'))

            # print out the 3d hand position
            cv2.putText(frame, f"Hand 3D: x={x:.5f}, y={y:.5f}, z={z:.5f}",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if time.time() - last_print_time > 1.5:
                # print(f"Hand (bounding box) 3D position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                last_print_time = time.time()


        now = time.time()
        if now - last_capture_time > capture_interval:
            crop = depth_frame[y1:y2, x1:x2]
            if crop.size > 0:
                
                # filename = f"/Users/rossgoldbaum/Desktop/CS1430_Projects/cv-final/captures/hand_{image_counter:04d}.png"
                # filename2 = f"/Users/rossgoldbaum/Desktop/CS1430_Projects/cv-final/captures/hand_{image_counter:04d}_2.png"
                depth_norm = np.clip((crop / 4500.0) * 255, 0, 255).astype(np.uint8)
                # cv2.imwrite(filename, depth_norm)
                depth_norm = match_training_style(depth_norm)
                classification = classify(depth_norm, clawModel)
                # print(classification)
                if classification == "closed":
                    influence = 1
                else: 
                    influence -= 0.25
                
                if influence > 0.3:
                    closeLabel = "closed!"
                    clawAngle = 5
                else: 
                    closeLabel = "open!"
                    clawAngle = 55
                last_capture_time = now
                
                # cv2.imwrite(filename2, depth_norm)
                image_counter+=1
                


        if len(keypoints_2d) == 6:
            input_tensor = torch.tensor([keypoints_2d], dtype=torch.float32)
            with torch.no_grad():
                pred_3d = model(input_tensor).view(3, 3)  # (3 joints, 3D)

            # Update history
            for i, joint in enumerate(pred_3d):
                history[i].append(joint.numpy())

            # 2D plot
            joint2d = np.array(keypoints_2d).reshape(3, 2)
            joint2d_line.set_data(joint2d[:, 0], joint2d[:, 1])

            # 3D plot
            # for i, line in enumerate(lines):
            #     data = np.array(history[i])
            #     if len(data) > 0:
            #         line.set_data(data[:, 0], data[:, 1])
            #         line.set_3d_properties(data[:, 2])

            # if all(len(h) > 0 for h in history):
            #     joints_xyz = [h[-1] for h in history]
            #     joints_xyz = np.stack(joints_xyz)
            #     skeleton_line.set_data(joints_xyz[:, 0], joints_xyz[:, 1])
            #     skeleton_line.set_3d_properties(joints_xyz[:, 2])


                # print out the coords every 1.5 seconds
                # if time.time() - last_print_time > 1.5:
                #     print("3D Arm Coordinates:")
                #     for label, pt in zip(["Shoulder", "Elbow", "Wrist"], joints_xyz):
                #         print(f"  {label}: x={pt[0]:+.3f}, y={pt[1]:+.3f}, z={pt[2]:+.3f}")
                #     print("-" * 40)
                #     last_print_time = time.time()

            plt.draw()
            plt.pause(0.001)
         
            
    
    cv2.imshow("Webcam - 2D Detection - "+closeLabel, frame)
    cv2.imshow("Depth View", depth_vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()