import os
import io
import logging
import threading
import time
import random
from queue import Queue
from collections import deque
from itertools import cycle

from PIL import Image
import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response, send_file

import av
logging.getLogger('libav').setLevel(logging.ERROR) #disable warnings from FFMPEG when processing rtsp streams

import detect_image

FPS=4 #frames per second in the MJPEG stream
PF=0.5 #percentage of frames processed from the rtsp stream 

CAMERA_URL = os.getenv('CAMERA', 'rtsp://admin@192.168.1.96:554/user=admin_password=_channel=0_stream=0.sdp')
MODEL_FILE = os.getenv('MODEL_FILE', "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
LABELS_FILE = os.getenv('LABELS_FILE', "models/coco_labels.txt")
MQTT_BASE_TOPIC = os.getenv('MQTT_BASE_TOPIC', 'cusca')
MQTT_SERVER = os.getenv('MQTT_SERVER', '192.168.1.100')
BUFFER_SIZE = os.getenv('BUFFER_SIZE', 24)

CONF_ARMED = 'armed'
CONF_MJPEG_FPS = 'mjpeg_fps'
CONF_PF = 'percentage_processed_frames'
CONF_THRESHOLD = 'threshold'
CONF_EVENT_BUFFER = 'buffer'

DBG_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=DBG_LEVEL, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class Camera(object):
    def __init__(self, url, callback = None):
        self.engine = detect_image.Engine(MODEL_FILE, LABELS_FILE )
        self.rtsp_url = url
        self.callback = callback
        self._event = False
        self.event_detected = False

        self.configuration = {
            CONF_ARMED: True,
            CONF_MJPEG_FPS: FPS,
            CONF_PF: PF, 
            CONF_THRESHOLD: detect_image.TF_THRESHOLD,
            CONF_EVENT_BUFFER: BUFFER_SIZE,
        }

        self.set_buffer()

    def set_buffer(self, buffer_size = BUFFER_SIZE):
        self.frames = Queue(maxsize=buffer_size)
        self.current_event = deque(maxlen=buffer_size)
        self.last_events = deque(maxlen=buffer_size)
        self.cycle = deque(maxlen=buffer_size)

    @property
    def event_detected(self):
        return self._event

    @event_detected.setter
    def event_detected(self, val):
        if self.callback and val != self._event:
            self.callback("event_detected", val)
        self._event = val

    def capture_frames(self):
        # This thread captures frames exclusively
        container = av.open(self.rtsp_url)
        container.streams.video[0].thread_type = 'AUTO'
        w, h = container.streams.video[0].width, container.streams.video[0].height
        fc = 0

        #Init with blank screen
        self.last_events.append(Image.new('RGB', (w, h)))
        self.cycle = self.last_events.copy()

        for frame in container.decode(video=0):
            fc = (fc+1)%(1/PF)
            if fc == 0:
                self.frames.put(frame)

    def process_frames(self):
        # This thread processes frames and can handle other tasks
        while True:
            frame = self.frames.get()
            img = frame.to_image()
            if self.configuration[CONF_ARMED]:
                d_img, prob = self.engine.detect_image(img, threshold=self.configuration[CONF_THRESHOLD])
            else:
                d_img = img
    
            if d_img:
                self.event_detected = True
                self.current_event.append(d_img)
                self.callback("event_probability", prob)
            else:
                self.event_detected = False

            if not self.event_detected and len(self.current_event) or \
                len(self.current_event) == self.current_event.maxlen: 
                
                self.last_events.extend(self.current_event)
                self.cycle = self.last_events.copy() #copy which we rotate
                self.current_event.clear()

    def get_frame(self):
        frame = self.cycle[0]
        self.cycle.rotate(-1)
        return frame
    
    def last_frame(self):
        frame = self.last_events[-1]
        return frame
    
app = Flask(__name__)

camera = Camera(CAMERA_URL)  

@app.route('/')
def index():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image')
def image():
    frame = camera.last_frame()
    imgByteArr = io.BytesIO()
    frame.save(imgByteArr, 'JPEG')
    imgByteArr.seek(0)

    return send_file(imgByteArr, mimetype='image/jpeg')

def gen():
    """Generate MJPEG frames."""
    while True:
        time.sleep(1/camera.configuration[CONF_MJPEG_FPS])
        frame = camera.get_frame()
        imgByteArr = io.BytesIO()
        frame.save(imgByteArr, format='JPEG')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + imgByteArr.getvalue() + b'\r\n')

def publish_property(client, property, value):
    client.publish(f"{MQTT_BASE_TOPIC}/{property}",value)

def on_connect(client, properties, flags, result):
    client.publish(MQTT_BASE_TOPIC+"/status","online",retain=True)
    for conf in camera.configuration:
        client.publish(f"{MQTT_BASE_TOPIC}/{conf}", camera.configuration[conf])
        client.subscribe(f"{MQTT_BASE_TOPIC}/{conf}", 0)

def on_message(client, userdata, message):
    logger.debug("on_message %s = %s", message.topic, str(message.payload))

    if CONF_EVENT_BUFFER is message.topic:
        if str(message.payload.decode()).isnumeric():
            camera.set_buffer(int(str(message.payload.decode())))
        else:
            logger.error(f"Could not set buffer size to {message.payload}")

    if CONF_ARMED in message.topic: 
        if "true" in str(message.payload).lower():
            userdata[CONF_ARMED] = True
        else:
            userdata[CONF_ARMED] = False
        logger.info("System is %s", "Armed" if userdata[CONF_ARMED] else "Disarmed")

    elif CONF_MJPEG_FPS in message.topic:
        if str(message.payload.decode()).isnumeric():
            userdata[CONF_MJPEG_FPS] = int(str(message.payload.decode()))
            logger.info(f"Setting MJPEG stream to {userdata[CONF_MJPEG_FPS]} FPS")
        else:
            logger.error(f"Could not set mjpeg_fps to {message.payload}")

    elif CONF_PF in message.topic:
        try:
            userdata[CONF_PF] = float(message.payload)
            logger.info(f"Setting % processed frames to {userdata[CONF_PF]*100}%")
        except ValueError:
            logger.error(f"Could not set percentage_processed_frames to {message.payload}")

    elif CONF_THRESHOLD in message.topic:
        try:
            userdata[CONF_THRESHOLD] = float(message.payload)
            logger.info(f"Setting threshold to {userdata[CONF_THRESHOLD]}")
        except ValueError:
            logger.error(f"Could not set threshold to {message.payload}")

if __name__ == '__main__':
    mqttc = mqtt.Client(client_id=f"cusca_{random.randint(10, 99)}", userdata=camera.configuration)
    mqttc.will_set(MQTT_BASE_TOPIC+"/status", "offline",retain=True)
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    mqttc.connect(MQTT_SERVER)
    mqttc.loop_start()
    
    camera.callback = lambda p, v: publish_property(mqttc, p, v)
    
    c = threading.Thread(target=camera.capture_frames)
    p = threading.Thread(target=camera.process_frames)
    c.start()
    p.start()
    app.run(host='0.0.0.0', debug=False)
    c.join()
    p.join()
    mqttc.loop_stop()