import os
import time
import cv2
import torch
import numpy as np


class BaseDetector:

    def __init__(self, target_model: str, model_name:str, person_class_id: int ):

        self.PERSON_CLASS_ID = person_class_id
        self.model_path = os.path.join("models", target_model, model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hyperparams = self._load_hyperparameters()
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        self.video = None
        self.video_name = None
        self.writer = None

        self.frame_times = []
        self.confidences = []
        self.people_count = []
        self.boxes_per_frame = []

        self.video_path = "source"

        self.output_video_path = os.path.join("result", target_model, "video")
        self.result_metrics_path = os.path.join("result", target_model, "metrics")

        os.makedirs(self.output_video_path, exist_ok=True)
        os.makedirs(self.result_metrics_path, exist_ok=True)

    def _load_hyperparameters(self):
        return {
            "conf_threshold": 0.5,
            "iou_threshold": 0.5
        }

    def load_video(self, video_name: str):
        self.video_name = video_name
        path = os.path.join(self.video_path, video_name)
        self.video = cv2.VideoCapture(path)

        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join(self.output_video_path, video_name)
        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

    def run(self, drawer):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            start = time.time()
            boxes = self.predict(frame)
            end = time.time()

            self.boxes_per_frame.append(boxes)
            self.frame_times.append(end - start)
            self.people_count.append(len(boxes))

            if boxes:
                self.confidences.extend([b[4] for b in boxes])

            frame = drawer.draw_boxes(frame, boxes)
            self.writer.write(frame)

        self.video.release()
        self.writer.release()

        self._save_metrics(drawer)

    def _save_metrics(self, drawer):
        metrics = {
            "FPS": np.mean([1 / t for t in self.frame_times if t > 0]),
            "Avg confidence": np.mean(self.confidences) if self.confidences else 0,
            "Max people per frame": max(self.people_count),
            "Avg people per frame": np.mean(self.people_count),
            "Self-IoU over time": self._self_iou_over_time()
        }

        drawer.draw_metrics(
            video_name=self.video_name,
            metrics=metrics,
            save_dir=self.result_metrics_path
        )

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _self_iou_over_time(self):
        ious = []

        for prev, curr in zip(self.boxes_per_frame[:-1], self.boxes_per_frame[1:]):
            if not prev or not curr:
                continue

            for b1 in prev:
                best_iou = 0
                for b2 in curr:
                    iou = self._iou(b1, b2)
                    best_iou = max(best_iou, iou)
                ious.append(best_iou)

        return sum(ious) / len(ious) if ious else 0

    def predict(self, frame):
        raise NotImplementedError

    def _load_model(self):
        raise NotImplementedError
