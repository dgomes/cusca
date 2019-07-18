import os
import io
import logging
import threading
import time
from collections import deque

import av
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response

FPS=2 #frames per second in the MJPEG stream
PF=8 #fraction of frames processed from the stream 1/PF

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
            logger.info("%s", self.engine.required_input_array_size())
        except Exception as err:
            logger.error(err)

    def get_frame(self):
        fnt = ImageFont.truetype('opensans.ttf', 20)
        video = av.open(self.rtsp_url)
        fps = 0
        for s in video.streams:
            if s.type == 'video':
                for packet in video.demux(s):
                    for frame in packet.decode():
                        fps = (fps+1)%PF
                        if fps == 0:
                            img = frame.to_image()
                            draw = ImageDraw.Draw(img)
                           
                            specials = [(1300, 100, 200), (1400, 600, 400)]
                            for r in specials:
                                reg = (r[0], r[1], r[0] + r[2], r[1] + r[2])
                                draw.rectangle(reg, outline='green')
                                crop_img = img.crop(reg)
                            
                                if self.engine:
                                    ans = self.engine.DetectWithImage(crop_img, threshold=0.05, keep_aspect_ratio=True, relative_coord=False, top_k=10)
                                    for obj in ans:
                                        box = obj.bounding_box.flatten().tolist()
                                        logger.debug("%s, %s, %s", box, obj.label_id, obj.score)
                                        draw.rectangle(box, outline='red')
                                        draw.text((int(box[0]), int(box[1])-24), str(obj.label_id), font=fnt, fill="red")

                            qcam.append(img)

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
        if len(qcam):
            frame = qcam.popleft()
            imgByteArr = io.BytesIO()
            frame.save(imgByteArr, format='JPEG')
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + imgByteArr.getvalue() + b'\r\n')

@app.route('/live')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    c = threading.Thread(target=camera.get_frame)
    c.start()
    app.run(host='0.0.0.0', debug=False)
    c.join()
