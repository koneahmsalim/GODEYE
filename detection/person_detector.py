"""Détection YOLO restreinte aux personnes (classe COCO 0)."""

from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path="models/yolov8s.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Retourne les détections au format attendu par DeepSORT."""
        detections = []
        for result in self.model(frame):
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id != 0:  # COCO 0 = person
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    (
                        [x1, y1, x2 - x1, y2 - y1],
                        float(box.conf[0]),
                        "person",
                    )
                )
        return detections
