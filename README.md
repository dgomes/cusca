# CUSCA - *CU*rrent event *S*treaming *C*amer*A*

This webapp generates an MJPEG Stream from an RTSP stream. The stream is based on filtered frames, with meaningful events (currently persons detected in the frame)

This webapp requires a Google Coral TPU.

# Install

- Install edgetpu libraries from https://coral.ai/software/
- Download tflite from https://www.tensorflow.org/lite/guide/python
- Adjust requirements.txt (to match your OS)

# Running

```python3 app```
