import os
import io
import logging
import threading
import time
from collections import deque
from itertools import cycle

import av
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response

import detect_image

FPS=2 #frames per second in the MJPEG stream
PF=8 #fraction of frames processed from the rtsp stream 1/PF

MODEL_FILE = os.getenv('MODEL_FILE', "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
LABELS_FILE = os.getenv('LABELS_FILE', "models/coco_labels.txt")

DBG_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=DBG_LEVEL, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class Camera(object):
    def __init__(self, url):
        self.engine = detect_image.Engine(MODEL_FILE, LABELS_FILE )
        self.rtsp_url = url
        self._initstate = True

        self.qcam = deque(maxlen=24)
        self.last_event = deque(maxlen=24)

    def capture_frames(self):
        rtsp = av.open(self.rtsp_url)
        fps = 0

        stream = rtsp.streams.video[0]
        for packet in rtsp.demux(stream):
            for frame in packet.decode():

                #populate initial image
                if self._initstate:
                    img = frame.to_image()
                    self.qcam.append(img)
                    self._initstate = False

                fps = (fps+1)%PF
                if fps == 0:
                    img = frame.to_image()
                    
                    d_img = self.engine.detect_image(img)
                    if d_img:
                        self.qcam.append(d_img)

    def get_frame(self):
        if len(self.qcam):
            frame = self.qcam.popleft()
            self.last_event.append(frame)
        else:
            frame = self.last_event.popleft()
            self.last_event.append(frame)
        return frame

app = Flask(__name__)

CAMERA_URL = os.getenv('CAMERA', 'rtsp://admin@192.168.1.96:554/user=admin_password=_channel=0_stream=0.sdp')
camera = Camera(CAMERA_URL)  

@app.route('/')
def index():
    return "pick a camera"

def gen():
    """Generate MJPEG frames."""
    while True:
        time.sleep(1/FPS)
        frame = camera.get_frame()
        imgByteArr = io.BytesIO()
        frame.save(imgByteArr, format='JPEG')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + imgByteArr.getvalue() + b'\r\n')
            

@app.route('/live')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    c = threading.Thread(target=camera.capture_frames)
    c.start()
    app.run(host='0.0.0.0', debug=False)
    c.join()
