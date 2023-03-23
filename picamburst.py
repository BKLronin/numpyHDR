#!//usr/bin/python3

from picamera2 import Picamera2
from libcamera import controls

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
picam2.set_controls({"AwbEnable": 1, "AeEnable": 1, "AeConstraintMode": controls.AeConstraintModeEnum.Highlight})
np_array = picam2.capture_array()
print(np_array)

picam2.stop()