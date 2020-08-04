from abc import ABC, abstractmethod

TF_THRESHOLD = 0.14

class Engine(ABC):
    def __init__(self, model_file):
        self._interesting_objs = [0] # 0 is person TODO filter from labels
        self.interpreter = None

        super().__init__()

    @abstractmethod
    def detect_image(self, image=None, image_file=None, threshold=TF_THRESHOLD):
        pass
 
    def draw_object(self, draw, bbox, label, score):
        """Draws the bounding box and label for an object."""
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10), f"{label}\n{score:.2f}",
                    fill='red')