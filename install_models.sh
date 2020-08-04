#!/bin/sh
mkdir -p models
cd models
curl -O https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/coco_labels.txt
curl -O https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar zxvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb .
rm -rf ssd_mobilenet_v2_coco_2018_03_29
rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ..
