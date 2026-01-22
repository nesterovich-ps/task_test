from ultralytics import YOLO
from .base_detector import BaseDetector


class YoloDetector(BaseDetector):
    def __init__(self):
        target_model = "yolo"
        model_name = "yolo11m.pt"
        person_class_id = 0
        super().__init__(target_model, model_name, person_class_id)

    def _load_model(self):
        return YOLO(self.model_path)

    def predict(self, frame):
        results = self.model(
            frame,
            conf=self.hyperparams["conf_threshold"],
            iou=self.hyperparams["iou_threshold"],
            classes=[self.PERSON_CLASS_ID],
            verbose=False
        )

        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf[0])
                boxes.append((x1, y1, x2, y2, score))

        return boxes
