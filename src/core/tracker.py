import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    from boxmot import BotSort
except ImportError:
    class BotSort:
        def __init__(self, **kwargs):
            raise ImportError(
                "boxmot package is not installed. Please install it with: pip install boxmot"
            )


class BotSortTracker:
    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "ecc",
        frame_rate: int = 30,
        device: str = "cpu",
        history_size: int = 50,
        reid_weights_path: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        assert 0.0 <= track_high_thresh <= 1.0, "track_high_thresh must be between 0 and 1"
        assert 0.0 <= track_low_thresh <= 1.0, "track_low_thresh must be between 0 and 1"
        assert 0.0 <= new_track_thresh <= 1.0, "new_track_thresh must be between 0 and 1"

        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.cmc_method = cmc_method
        self.frame_rate = frame_rate

        self.reid_weights_path = self._resolve_weights_path(reid_weights_path)

        self.tracker = self._build_tracker()

        self.track_history: dict[int, deque[tuple[float, float]]] = {}
        self.history_size = max(1, history_size)

    def _resolve_weights_path(
        self, reid_weights_path: str | Path | None
    ) -> Path | None:
        if not reid_weights_path:
            return None

        candidate = Path(reid_weights_path)
        return candidate if candidate.exists() else None

    def _build_tracker(self) -> BotSort:
        weights = self.reid_weights_path or Path("models/osnet_x0_25_msmt17.pt")

        return BotSort(
            reid_weights=weights,
            device=self.device,
            half=False,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            cmc_method=self.cmc_method,
            frame_rate=self.frame_rate,
        )

    def update(self, detections: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
        boxes = detections.get("boxes", [])
        confidences = detections.get("confidences", [])
        class_ids = detections.get("class_ids", [])

        if len(boxes) == 0:
            return {
                "boxes": [],
                "confidences": [],
                "track_ids": [],
                "class_ids": [],
                "active_ids": set(),
                "track_positions": {},
                "track_confidences": {},
            }

        dets = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            cls_id = class_ids[i]
            dets.append([x1, y1, x2, y2, conf, cls_id])

        dets = np.array(dets)

        tracked_objects = self.tracker.update(dets, frame)

        tracked_boxes = []
        tracked_confidences = []
        tracked_ids = []
        tracked_class_ids = []
        track_positions = {}
        track_confidences = {}

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, conf, cls_id = obj[:7]

            tracked_boxes.append([x1, y1, x2, y2])
            tracked_confidences.append(conf)
            track_id_int = int(track_id)
            tracked_ids.append(track_id_int)
            tracked_class_ids.append(int(cls_id))

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            track_positions[track_id_int] = (center_x, center_y)
            track_confidences[track_id_int] = float(conf)

            history = self.track_history.setdefault(
                track_id_int, deque(maxlen=self.history_size)
            )
            history.append((center_x, center_y))

        return {
            "boxes": np.array(tracked_boxes),
            "confidences": np.array(tracked_confidences),
            "track_ids": np.array(tracked_ids),
            "class_ids": np.array(tracked_class_ids),
            "active_ids": set(tracked_ids),
            "track_positions": track_positions,
            "track_confidences": track_confidences,
        }

    def get_track_history(self, track_id: int) -> list[tuple[float, float]]:
        history = self.track_history.get(track_id)
        if history is None:
            return []
        return list(history)

    def reset(self) -> None:
        self.track_history.clear()
        self.tracker = self._build_tracker()

