from PIL import Image
from PIL import ImageDraw

import tflite_runtime.interpreter as tflite
import platform
import os
import logging

import detect

TF_THRESHOLD = 0.14

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

DBG_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=DBG_LEVEL, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class Engine():

    def __init__(self, model_file, labels, threshold=TF_THRESHOLD):
        self._labels = self.load_labels(labels) if labels else {}

        self._interesting_objs = [0] # 0 is person TODO filter from labels

        # Load interpreter
        model_file, *device = model_file.split('@')
        self.interpreter = tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
            ])

        self.interpreter.allocate_tensors()

    def load_labels(self, path, encoding='utf-8'):
        """Loads labels from file (with or without index numbers).

        Args:
            path: path to label file.
            encoding: label file encoding.
        Returns:
            Dictionary mapping indices to labels."""
        with open(path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            if not lines:
                return {}

            if lines[0].split(' ', maxsplit=1)[0].isdigit():
                pairs = [line.split(' ', maxsplit=1) for line in lines]
                return {int(index): label.strip() for index, label in pairs}
            else:
                return {index: line.strip() for index, line in enumerate(lines)}
 
    def draw_object(self, draw, obj):
        """Draws the bounding box and label for an object."""
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                '%s\n%.2f' % (self._labels.get(obj.id, obj.id), obj.score),
                fill='red')
 
    def detect_image(self, image=None, image_file=None):
        if image_file:
            image = Image.open(image_file)
        scale = detect.set_input(self.interpreter, image.size,
                                lambda size: image.resize(size, Image.ANTIALIAS))

        self.interpreter.invoke()
        objs = detect.get_output(self.interpreter, TF_THRESHOLD, scale) 
        interesting_objs = [o for o in objs if o.id in self._interesting_objs]

        logger.debug(f"{len(objs)} objects detected - {len(interesting_objs)} of interest")

        for obj in interesting_objs:
            logger.debug(f"{self._labels.get(obj.id, obj.id)} detected with probabiliy {obj.score} at {obj.bbox}")
        
        if len(interesting_objs):
            image = image.convert('RGB')
            best_obj = max(interesting_objs, key=lambda x: x.score)
            self.draw_object(ImageDraw.Draw(image), best_obj)
            logger.info(f"{self._labels.get(best_obj.id, best_obj.id)} detected with probabiliy {best_obj.score} at {best_obj.bbox}")
            return image
        
        return None

if __name__ == '__main__':
    e = Engine()
    e.detect_image(image_file="19-49-38.jpg")