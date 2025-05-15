import serial
import time

ser = serial.Serial('/dev/tty.usbmodem1201', 9600, timeout=1)
time.sleep(2)

while True:
    angles = [160, 40, 100, 30]
    
    angle_str = 'START,' + ','.join(map(str, angles)) + '\n'
    ser.write(angle_str.encode('utf-8'))

ser.close()

