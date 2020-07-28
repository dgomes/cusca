#!/bin/sh
mkdir -p models
cd models
curl -O https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/coco_labels.txt
curl -O https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
cd ..
