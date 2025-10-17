from typing import Any

import cv2
import numpy as np


def resize_frame_to_fit(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
    maintain_aspect_ratio: bool = True,
) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    original_height, original_width = frame.shape[:2]

    if maintain_aspect_ratio:
        scale = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        return cv2.resize(frame, (new_width, new_height))
    else:
        return cv2.resize(frame, (max_width, max_height))


class Visualizer:
    def __init__(
        self,
        show_confidence: bool = True,
        show_track_id: bool = True,
        show_trajectory: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2,
        trajectory_length: int = 50,
    ) -> None:
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id
        self.show_trajectory = show_trajectory
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.trajectory_length = trajectory_length

        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
            (0, 128, 255),
            (255, 128, 128),
        ]

    def _get_color(self, track_id: int) -> tuple[int, int, int]:
        return self.colors[track_id % len(self.colors)]

    def draw_tracking_info(
        self, frame: np.ndarray, tracking_results: dict[str, Any]
    ) -> np.ndarray:
        boxes = tracking_results.get("boxes", [])
        confidences = tracking_results.get("confidences", [])
        track_ids = tracking_results.get("track_ids", [])
        track_positions = tracking_results.get("track_positions", {})

        for i, (box, conf, track_id) in enumerate(zip(boxes, confidences, track_ids)):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            color = self._get_color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            if track_id in track_positions:
                center_x, center_y = track_positions[track_id]
                cv2.circle(frame, (int(center_x), int(center_y)), 5, color, -1)

            if self.show_track_id:
                label = f"ID: {track_id}"
                if self.show_confidence:
                    label += f" ({conf:.2f})"

                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 5),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    self.font_thickness,
                )

        return frame

    def draw_trajectory(
        self,
        frame: np.ndarray,
        tracking_results: dict[str, Any],
        track_histories: dict[int, list[tuple[float, float]]],
    ) -> np.ndarray:
        if not self.show_trajectory:
            return frame

        track_ids = tracking_results.get("track_ids", [])

        for track_id in track_ids:
            track_id = int(track_id)

            if track_id not in track_histories:
                continue

            trajectory = track_histories[track_id]

            if len(trajectory) > self.trajectory_length:
                trajectory = trajectory[-self.trajectory_length :]

            color = self._get_color(track_id)

            for i in range(1, len(trajectory)):
                pt1 = (int(trajectory[i - 1][0]), int(trajectory[i - 1][1]))
                pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))

                alpha = i / len(trajectory)
                line_color = tuple(int(c * alpha) for c in color)

                cv2.line(frame, pt1, pt2, line_color, 2)

            for point in trajectory:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)

        return frame

    def draw_student_count(self, frame: np.ndarray, count_info: dict[str, Any]) -> np.ndarray:
        current_count = count_info.get("current_count", 0)
        total_unique = count_info.get("total_unique_students", 0)
        max_concurrent = count_info.get("max_concurrent_students", 0)

        height, width = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_offset = 35
        line_height = 25

        count_text = f"Students Present: {current_count}"
        cv2.putText(
            frame,
            count_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
        )

        y_offset += line_height

        unique_text = f"Total Unique: {total_unique}"
        cv2.putText(
            frame,
            unique_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        y_offset += line_height

        max_text = f"Max Concurrent: {max_concurrent}"
        cv2.putText(
            frame,
            max_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        fps_text = f"FPS: {fps:.1f}"

        height, width = frame.shape[:2]
        text_x = width - 150
        text_y = 30

        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        cv2.rectangle(
            frame,
            (text_x - 10, text_y - text_height - 10),
            (text_x + text_width + 10, text_y + 10),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            fps_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame
