from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
import logging
import os
from operator import itemgetter

import collections as col
#create employee NamedTuple
Object = col.namedtuple('Object', ['id', 'score', 'bbox'])
Box = col.namedtuple('Box', ['ymin', 'xmin', 'ymax', 'xmax'])

import detect
from engine import Engine, TF_THRESHOLD

DBG_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=DBG_LEVEL, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class CPUEngine(Engine):
    def __init__(self, model_file):
        super().__init__(model_file)

        self.__detection_graph = tf.Graph()
        with self.__detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0}
        )
        self.__sess = tf.compat.v1.Session(
            graph=self.__detection_graph,
            config=config)

    def detect_image(self, image=None, image_file=None, threshold=TF_THRESHOLD):
        if image_file:
            image = Image.open(image_file)
        
        ops = self.__detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.__detection_graph.get_tensor_by_name(tensor_name)

        image_tensor = self.__detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = self.__sess.run(tensor_dict,
                                      feed_dict={image_tensor: np.expand_dims(image, axis=0)})

        label_codes = output_dict['detection_classes'][0]
        scores = output_dict['detection_scores'][0]
        boxes = output_dict['detection_boxes'][0]

        interesting_objs = [e for e,s in enumerate(scores) if s>TF_THRESHOLD] # and label_codes[e] in self._interesting_objs]

        logger.debug(f"{len(scores)} objects detected - {len(interesting_objs)} of interest")

        for obj in interesting_objs:
            logger.debug(f"{label_codes[obj]} detected with probabiliy {scores[obj]} at {boxes[obj]}")
              
        if len(interesting_objs):
            image = image.convert('RGB')
            best_obj = max(interesting_objs, key=lambda x: scores[x])
            box = Box( int(boxes[best_obj][0] *image.height), int(boxes[best_obj][1]*image.width),
                       int(boxes[best_obj][2]*image.height), int(boxes[best_obj][3]*image.width))
            self.draw_object(ImageDraw.Draw(image), box, label_codes[best_obj], scores[best_obj])
            logger.info(f"{label_codes[best_obj]} detected with probabiliy {scores[best_obj]} at {boxes[best_obj]}")
            return image, float(scores[best_obj])
        
        return None, 0

if __name__ == '__main__':
    e = CPUEngine("models/frozen_inference_graph.pb")
    d,_ = e.detect_image(image_file="19-49-38.jpg", threshold=0)
    d.save("lixo.jpg")