#!//usr/bin/python3

import numpyHDR
from picamera2 import Picamera2
from libcamera import controls

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
ctrls = Controls(picam2)
ctrls.AwbEnable = 1
ctrls.AeEnable = 1
ctrls.ConstraintModeEnum.Highlight
np_array_ev0 = picam2.capture_array()
#picam2.set_controls({"AwbEnable": 1, "AeEnable": 1, "AeConstraintMode": controls.AeConstraintModeEnum.Shadows})
#np_array_ev1 = picam2.capture_array()
#picam2.set_controls({"AwbEnable": 1, "AeEnable": 1, "AeConstraintMode": controls.AeConstraintModeEnum.Normal})
#np_array_ev_neg1 = picam2.capture_array()
meta = "nichts"
picam2.helpers.save(np_array_ev0, meta, "1.jpg")
#picam2.helpers.save(np_array_ev1, meta ,"2.jpg")
#picam2.helpers.save(np_array_ev_neg1, meta, "3.jpg")

picam2.stop()