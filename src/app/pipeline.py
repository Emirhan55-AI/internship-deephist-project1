from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.utils.visualization import Visualizer, resize_frame_to_fit


class FrameProcessor:
    def __init__(
        self,
        detector,
        tracker,
        counter,
        visualizer: Visualizer,
        resize_config: dict[str, Any],
        visualization_config: dict[str, Any],
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.visualizer = visualizer

        self.resize_enabled = resize_config.get("enabled", False)
        self.resize_width = resize_config.get("width", 0)
        self.resize_height = resize_config.get("height", 0)
        self.maintain_aspect = resize_config.get("maintain_aspect_ratio", True)

        self.show_trajectory = visualization_config.get("show_trajectory", True)
        
        self.last_tracking: dict[str, Any] = {}
        self.last_counts: dict[str, Any] = {}

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        if frame is None:
            raise ValueError("FrameProcessor.process received a None frame")

        working_frame = self._maybe_resize(frame)

        detections = self.detector.detect(working_frame)
        tracking = self.tracker.update(detections, working_frame)
        counts = self.counter.update_count(tracking)

        self.last_tracking = tracking
        self.last_counts = counts

        annotated_frame = self._draw_visuals(working_frame, tracking, counts)

        return {
            "frame": annotated_frame,
            "tracking": tracking,
            "counts": counts,
        }
    
    def process_skip(self, frame: np.ndarray) -> dict[str, Any]:
        if frame is None:
            raise ValueError("FrameProcessor.process_skip received a None frame")

        working_frame = self._maybe_resize(frame)

        annotated_frame = self._draw_visuals(working_frame, self.last_tracking, self.last_counts)

        return {
            "frame": annotated_frame,
            "tracking": self.last_tracking,
            "counts": self.last_counts,
        }

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        if not self.resize_enabled:
            return frame
        if not self.resize_width or not self.resize_height:
            return frame
        return resize_frame_to_fit(
            frame,
            max_width=self.resize_width,
            max_height=self.resize_height,
            maintain_aspect_ratio=self.maintain_aspect,
        )

    def _draw_visuals(
        self,
        frame: np.ndarray,
        tracking_results: dict[str, Any],
        counts: dict[str, Any],
    ) -> np.ndarray:
        annotated = self.visualizer.draw_tracking_info(frame, tracking_results)

        if self.show_trajectory:
            histories = self._collect_histories(tracking_results)
            if histories:
                annotated = self.visualizer.draw_trajectory(annotated, tracking_results, histories)

        annotated = self.visualizer.draw_student_count(annotated, counts)
        return annotated

    def _collect_histories(
        self, tracking_results: dict[str, Any]
    ) -> dict[int, Iterable[tuple[float, float]]]:
        active_ids: set[int] = set()
        raw_ids = tracking_results.get("active_ids", set())

        if isinstance(raw_ids, np.ndarray):
            active_ids = {int(item) for item in raw_ids.tolist()}
        elif isinstance(raw_ids, (list, set, tuple)):
            active_ids = {int(item) for item in raw_ids}

        histories: dict[int, Iterable[tuple[float, float]]] = {}
        for track_id in active_ids:
            histories[track_id] = self.tracker.get_track_history(track_id)
        return histories
