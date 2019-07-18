import os
import io
import logging
import threading
import time
from collections import deque

import av
from PIL import Image, ImageDraw

from flask import Flask, render_template, Response

DBG_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=DBG_LEVEL, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

qcam = deque(maxlen=24)

class Camera(object):
    def __init__(self, url):
        self.engine = None
        self.rtsp_url = url
        try:
            from edgetpu.detection.engine import DetectionEngine

            # Initialize engine.
            self.engine = DetectionEngine("mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
        except:
            logger.error("edgetpu libs not installed")

    def get_frame(self):
        video = av.open(self.rtsp_url)
        fps = 0
        for s in video.streams:
            if s.type == 'video':
                for packet in video.demux(s):
                    for frame in packet.decode():
                        fps = (fps+1)%8
                        if fps == 0:
                            img = frame.to_image()
                            if self.engine:
                                ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True, relative_coord=False, top_k=10)
                                for obj in ans:
                                    print(obj.label_id, obj.score)
                            qcam.append(img)

app = Flask(__name__)
camera = Camera("rtsp://admin@192.168.1.96:554/user=admin_password=_channel=0_stream=0.sdp")  

@app.route('/')
def index():
    return "pick a camera"

def gen():
    """Generate a JPEG frame."""
    while True:
        time.sleep(0.5)
        if len(qcam):
            print(len(qcam))
            frame = qcam.popleft()
            imgByteArr = io.BytesIO()
            frame.save(imgByteArr, format='JPEG')
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + imgByteArr.getvalue() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    c = threading.Thread(target=camera.get_frame)
    c.start()
    app.run(host='0.0.0.0', debug=True)
    c.join()