# Kinect-Controlled Robotic Arm

NOTE: Only the weights of the CNN are captured here, for the sake of space, the training data is excluded.
ANOTHER NOTE: This is not very possible to run on its own, the responses must be sent to a serial output and requires a Kinect to run. 
## Overview

This project combines computer vision, deep learning, and robotics to build an end-to-end system where a user's hand movements, captured by an Xbox 360 depth camera, are translated in real time to a servo-driven robotic arm!!!

- `main.py` Contains the main run-loop of the Kinect stream and processing, data sent over the serial connection, and 
- `arm_interpretation.ino` Contains the signal handling code for the Arduino, which updates continuously even when separate from serial data

---

## System Architecture

### Input: Depth & RGB Sensing

- An **Xbox 360 Kinect camera** captures both:
  - A full-color RGB stream
  - A synchronized depth map (in millimeters)
- The **RGB stream** is used for 2D pose estimation
- The **depth map** provides 3D spatial data

### Joint Detection and Hand Localization

- We use **MediaPipe Pose** to identify key joints (elbow, wrist, etc.)
- From these joints, we:
  - Estimate the direction of the forearm
  - Predict the location of the user’s **hand**
  - Generate a **bounding box** around the hand for further analysis

### Hand Classification

- The bounding box is cropped from the **depth map**
- The resulting depth image is normalized and preprocessed
- A **Convolutional Neural Network (CNN)** classifies the hand as:
  - **Open**
  - **Closed**

### 3D Hand Position Estimation

- The depth within the hand bounding box is analyzed
- We extract the **average depth** of the closest valid pixels
- Using intrinsic camera parameters (fx, fy, cx, cy), we estimate:
  - **X, Y, Z coordinates** of the hand in space

### Robotic Arm Control

- The 3D hand position and open/closed state are encoded into a serial message
- An **Arduino** receives this message over USB
- The position is interpreted through **inverse kinematics**, solving for:
  - **Base rotation**
  - **Shoulder elevation**
  - **Elbow extension**
- A **custom-built servo-powered robotic arm** mirrors the hand’s position and gesture in real time

---

## Features

- Real-time hand tracking and control
- Depth-aware gesture detection
- Custom inverse kinematics mapping
- Integrated hardware/software pipeline
- Compatible with MG996R and micro servos

---

## Software Used

- Python + OpenCV
- MediaPipe Pose
- TensorFlow/Keras (CNN)
- Xbox Kinect (360 model)
- Arduino (servo control via PWM)
- Custom laser-cut arm structure

---

## Project Members

Johnny Kuehnis, Isabella Szabo, and Ross Goldbaum

