#!//usr/bin/python3

import time

from libcamera import controls
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.set_controls({"AwbEnable": 1})
picam2.set_controls({"AeEnable": 1})
picam2.set_controls({"AfMode": controls.AfModeEnum.Manual })
picam2.set_controls({"LensPosition": 0.0 })

def get_exposure_stack(factor: int = 2) -> list:
    '''Returns a list with arrays that contain different exposures controlled by the factor.
    The Autoamtically set exposure of the first frame is saved and multiplied or divided by the factor to get the above or under epxosures.

    :param factor: The EV difference in the exposure of the desired images
    :return: List containing arrays with the pictures
    '''


    picam2.start()
    time.sleep(1)

    print(picam2.capture_metadata())
    start = picam2.capture_metadata()
    exposure_start = start["ExposureTime"]
    gain_start = int(round(start["AnalogueGain"]))
    if exposure_start > 117000 or exposure_start < factor:
        print("Low Light levels out of range, can't make HDR")
        exit()

    picam2.set_controls({"AeEnable": 0})
    confirmed = picam2.capture_metadata()["AeLocked"]
    while confirmed != True:
        confirmed = picam2.capture_metadata()["AeLocked"]
        time.sleep(.1)

    picam2.set_controls({"AnalogueGain": gain_start})
    confirmed = picam2.capture_metadata()["AnalogueGain"]
    while confirmed != gain_start in range(gain_start -1, gain_start +1):
        confirmed = picam2.capture_metadata()["AnalogueGain"]
        time.sleep(.1)

    ev1 = picam2.capture_array()
    #print("Picture one is done")

    ev_low = int(exposure_start / factor)
    picam2.set_controls({"ExposureTime": ev_low})
    confirmed = picam2.capture_metadata()["ExposureTime"]
    while confirmed not in range(ev_low -100, ev_low + 100 ):
        confirmed = picam2.capture_metadata()["ExposureTime"]
        time.sleep(.01)

    #print("2",confirmed)
    ev2 = picam2.capture_array()
    #print("Picture 2 is captured to array")

    ev_high = int(exposure_start * factor)
    picam2.set_controls({"ExposureTime": ev_high})
    confirmed = picam2.capture_metadata()["ExposureTime"]
    while confirmed not in range(ev_high -100, ev_high + 100 ):
        confirmed = picam2.capture_metadata()["ExposureTime"]
        time.sleep(.01)

    #print("3",confirmed)
    ev3 = picam2.capture_array()
    #print("Picture 3 is captured")

    picam2.stop()
    stack = [ev1,ev2,ev3]

    return stack









