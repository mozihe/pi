from picamera2 import Picamera2
from libcamera import controls


class ImageController:
    def __init__(self):

        self.cam = Picamera2()
        self.cam.preview_configuration.main.size = (640, 360)
        self.cam.preview_configuration.main.format = "RGB888"
        self.cam.preview_configuration.controls.FrameRate = 50
        self.cam.preview_configuration.align()
        self.cam.configure("preview")
        self.cam.start()

    def getImg(self):
        try:
            frame = self.cam.capture_array()
            if frame is not None:
                return True, frame
            else:
                return False, None
        except Exception as e:
            print(f"Error capturing image: {e}")
            return False, None

    def __del__(self):
        self.cam.stop()
        self.cam.close()
