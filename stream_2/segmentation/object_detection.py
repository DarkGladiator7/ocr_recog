from yolov9.models.common import AutoShape
from yolov9.models.yolo import DetectionModel


class ObjectDetection:
    def __init__(self, img_size=640):
        self.model = None
        self.img_size = img_size

    def set_model(self, model_state_dict=None, conf=0.05, iou=0.05):
        cfg = 'src/resume_parser/models/yolo/yolov9-c.yaml'
        model = DetectionModel(cfg=cfg)

        model.load_state_dict(state_dict=model_state_dict, strict=False)
        model = AutoShape(model)

        model.conf = conf
        model.iou = iou

        self.model = model

    def predict(self, image):
        results = self.model(image, size=self.img_size)
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        return boxes

    def bulk_predict(self, images):
        results = self.model(images, size=self.img_size)
        img_wise_bboxes = [prediction[:, :4] for prediction in results.pred]
        return img_wise_bboxes
