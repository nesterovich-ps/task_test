import os
import torch
import torchvision
from torchvision.transforms import functional as F
from .base_detector import BaseDetector


class FasterRCNNDetector(BaseDetector):
    def __init__(self):
        target_model = "faster_rcnn"
        model_name = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
        person_class_id = 1
        super().__init__(target_model, model_name, person_class_id)

    def _load_model(self):

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if not os.path.exists(self.model_path):
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            torch.save(model.state_dict(), self.model_path)
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)

        return model

    @torch.no_grad()
    def predict(self, frame):
        image = F.to_tensor(frame).to(self.device)
        output = self.model([image])[0]

        result = []
        for box, label, score in zip(
                output["boxes"],
                output["labels"],
                output["scores"]
        ):
            if label.item() == self.PERSON_CLASS_ID and score >= self.hyperparams["conf_threshold"]:
                x1, y1, x2, y2 = map(int, box.tolist())
                result.append((x1, y1, x2, y2, float(score)))

        return result
