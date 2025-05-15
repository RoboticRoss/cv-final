import freenect
import numpy as np
import cv2
import serial
import time

arduino = serial.Serial('/dev/tty.usbmodem11301', 9600)
time.sleep(2)

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.uint8)

while True:
    depth = get_depth()

    small = cv2.resize(depth, (160, 120))
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(small)
    x, y = max_loc  # coordinates in the small image
    z = small[y, x]  # depth value (0–255)

    x_full = int(x * 640 / 160)
    y_full = int(y * 480 / 120)

    print(f"3D Point: X={x_full}, Y={y_full}, Z={z}")

    msg = f"{x_full},{y_full},{z}\n"
    arduino.write(msg.encode())

    cv2.circle(small, (x, y), 4, (255), -1)
    cv2.imshow("Depth", small)
    if cv2.waitKey(1) == 27:
        break
