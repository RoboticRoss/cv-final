import freenect
import cv2
import numpy as np
import serial
import time

# === CONFIG ===
SERIAL_PORT = '/dev/tty.usbmodem11301'  # change as needed
BAUD_RATE = 9600

# === Serial Setup ===
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
last_send = time.time()

smoothed_x, smoothed_y, smoothed_z = 0, 0, 0
alpha = 0.7  # (closer to 1 = smoother)

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.uint16)  # 16-bit millimeter precision

# def find_hand_3D(depth_frame):
#     # Mask for close-range values likely to be a hand
#     mask = (depth_frame > 300) & (depth_frame < 800)
#     masked = np.uint8(mask * 255)

#     contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest = max(contours, key=cv2.contourArea)
#         M = cv2.moments(largest)
#         if M["m00"] > 0:
#             cx = int(M["m10"] / M["m00"])  # X centroid
#             cy = int(M["m01"] / M["m00"])  # Y centroid
#             z = int(depth_frame[cy, cx])   # Depth at centroid
#             return cx, cy, z
#     return None

def normalize_to_byte(val, max_val):
    val = np.clip(val, 0, max_val)
    return int((val / max_val) * 255)

while True:
    depth = get_depth()
    result = find_hand_3D(depth)

    if time.time() - last_send > 0.1:
        if result is not None:
            x, y, z = result
            smoothed_x = alpha * smoothed_x + (1 - alpha) * x
            smoothed_y = alpha * smoothed_y + (1 - alpha) * y
            smoothed_z = alpha * smoothed_z + (1 - alpha) * z
            x_byte = normalize_to_byte(smoothed_x, depth.shape[1])  # 0–640
            y_byte = normalize_to_byte(smoothed_y, depth.shape[0])  # 0–480
            z_byte = normalize_to_byte(smoothed_z, 1024)            # assume max 1024mm for depth

            msg = f"{x_byte},{y_byte},{z_byte}\n"
            arduino.write(msg.encode())
            print("TO ARDUINO:", msg.strip())
        else:
            arduino.write(b"0,0,0\n")

        # Optional visual debug
        debug_vis = cv2.cvtColor((depth / 1024 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if result is not None:
            x, y, z = result

            # Redraw the mask and contours for the debug view
            mask = (depth > 300) & (depth < 800)
            masked = np.uint8(mask * 255)
            contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x_, y_, w, h = cv2.boundingRect(largest)
                cv2.rectangle(debug_vis, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 2)
                cv2.circle(debug_vis, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow("Depth Mask + Tracker", debug_vis)
        if cv2.waitKey(1) == 27:  # Escape key = exit
            break

arduino.close()
cv2.destroyAllWindows()
