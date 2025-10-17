from typing import Any

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        model_path: str = "models/yolo11s.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.3,
        device: str | None = None,
        classes: list[int] | None = None,
    ) -> None:
        assert 0.0 <= confidence_threshold <= 1.0, "confidence_threshold must be between 0 and 1"
        assert 0.0 <= iou_threshold <= 1.0, "iou_threshold must be between 0 and 1"

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or [0]
        self.device = device or "cpu"

        self.model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"YOLO model file not found at {self.model_path}. "
                f"Please ensure the model file exists."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from {self.model_path}: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot perform detection.")

        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided to detector")

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                device=self.device,
                verbose=False,
            )

            result = results[0]

            detections: dict[str, Any] = {
                "boxes": [],
                "confidences": [],
                "class_ids": [],
                "class_names": [],
                "raw_result": result,
            }

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                class_names = [result.names[cls_id] for cls_id in class_ids]

                detections["boxes"] = boxes
                detections["confidences"] = confidences
                detections["class_ids"] = class_ids
                detections["class_names"] = class_names

            return detections

        except Exception as exc:
            raise RuntimeError(f"Detection failed: {exc}") from exc

