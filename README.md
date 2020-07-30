# CUSCA - CUrrent event Streaming CamerA

This webapp generates an MJPEG Stream from an RTSP stream. The stream is based on filtered frames, with meaningful events (currently persons detected in the frame) that loop indefinentely until new events come in.

This webapp requires a Google Coral TPU.

# Install

- Install edgetpu libraries from https://coral.ai/software/
- Download tflite from https://www.tensorflow.org/lite/guide/python
- Adjust requirements.txt (to match your OS)

# Running

```python3 app```

# Configuration

All config options are based on ENV variables (easy to use with containers)

options available:

| Variable | Information | Default |
| ------------- | ------------- | ------------- | 
| CAMERA_URL | RTSP stream | 'rtsp://admin@192.168.1.96:554/user=admin_password=_channel=0_stream=0.sdp' | 
| MODEL_FILE | Path to file containing model to be used | "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" | 
| LABELS_FILE | Path to file containing labels to be used (matched with the model file) | "models/coco_labels.txt" | 
| MQTT_SERVER | MQTT Broker address | '192.168.1.100' |
| MQTT_BASE_TOPIC | Topic under which camera configuration and events are published | 'cusca' |
| BUFFER_SIZE | Number of frames kept by the MJPEG stream | 24 |

Furthermore, some variables can be adjusted in realtime through MQTT such as:

| Topic | Information | Default |
| ------------- | ------------- | ------------- | 
| cusca/armed | currently unused | True | 
| cusca/mjpeg_fps | frames per second in the mjpeg stream | 4 | 
| cusca/percentage_processed_frames | % or frames from the original rtsp - default 50% | 0.5 | 
| cusca/threshold | consider an event if probability higher then (0 to 1) | 0.14 |
| cusca/buffer | Number of frames kept by the MJPEG stream|  24 |
