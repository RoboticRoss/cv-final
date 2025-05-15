import serial
import time

# !!!! Change this depending on port
port = '/dev/tty.usbmodem1301'
ser = serial.Serial(port, 9600, timeout=1)
time.sleep(2) 


for i in range(30):
    ser.write(b'1') 
    time.sleep(.1)
    ser.write(b'0') 
    time.sleep(.1)


ser.write(b'1') 
time.sleep(1)
ser.write(b'0')  
time.sleep(1)
ser.write(b'1') 
time.sleep(1)



ser.close()