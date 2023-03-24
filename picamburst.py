#!//usr/bin/python3

import numpyHDR
from PIL import Image
from picamera2 import Picamera2
from libcamera import controls
import time

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.set_controls({"AwbEnable": 1})
picam2.set_controls({"AeEnable": 1})
#picam2.set_controls({"AnalogueGain": 1.0})
picam2.start()
time.sleep(1)

print(picam2.capture_metadata())
start = picam2.capture_metadata()
exposure_start = start["ExposureTime"]
gain_start = start["AnalogueGain"]


ev1 = picam2.capture_array()
print("Picture one is done")

picam2.set_controls({"AeEnable": 0})
confirmed = picam2.capture_metadata()["AeLocked"]
while confirmed != True:
	confimed = picam2.capture_metadata()["AeLocked"]
	time.sleep(.1)


picam2.set_controls({"AnalogueGain": gain_start})
confirmed = picam2.capture_metadata()["AnalogueGain"]
while confirmed != gain_start in range(gain_start -0.1, gain_start +0.1):
        confimed = picam2.capture_metadata()["AnalogueGain"]
        time.sleep(.1)

ev_low = int(exposure_start / 4)
picam2.set_controls({"ExposureTime": ev_low})
confirmed = picam2.capture_metadata()["ExposureTime"]
while confirmed not in range(ev_low -100, ev_low + 100 ):
        confirmed = picam2.capture_metadata()["ExposureTime"]
        time.sleep(.01)

print("2",confirmed)
ev2 = picam2.capture_array()
print("Picture 2 is captured to array")

ev_high = int(exposure_start * 4)
picam2.set_controls({"ExposureTime": ev_high})
confirmed = picam2.capture_metadata()["ExposureTime"]
while confirmed not in range(ev_high -100, ev_high + 100 ):
        confirmed = picam2.capture_metadata()["ExposureTime"]
        time.sleep(.01)

print("3",confirmed)
ev3 = picam2.capture_array()
print("Picture 3 is captured")
print("Saving..")

image = Image.fromarray(ev1)
image.save(f"test_hdr0.jpg", quality=50)

image = Image.fromarray(ev2)
image.save(f"test_hdr1.jpg", quality=50)

image = Image.fromarray(ev3)
image.save(f"test_hdr2.jpg", quality=50)

picam2.stop()
