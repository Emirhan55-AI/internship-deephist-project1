"""
Visualization Module

This module provides visualization utilities for drawing bounding boxes, track IDs,
trajectories, and statistics on video frames. It includes the Visualizer class
and helper functions for frame resizing.

The visualizer supports customizable drawing styles and can display:
- Bounding boxes around detected students
- Track IDs and confidence scores
- Movement trajectories (trails)
- Student count overlays
- FPS display

Example:
    >>> visualizer = Visualizer(
    ...     show_confidence=True,
    ...     show_track_id=True,
    ...     show_trajectory=True
    ... )
    >>> annotated_frame = visualizer.draw_tracking_info(frame, tracking_results)
"""

from typing import Any

import cv2
import numpy as np


def resize_frame_to_fit(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
    maintain_aspect_ratio: bool = True,
) -> np.ndarray:
    """Resize frame to fit within maximum dimensions while optionally maintaining aspect ratio.
    
    This function resizes a video frame to fit within specified maximum width and height.
    If maintain_aspect_ratio is True, the frame is scaled proportionally to fit within
    the bounds without distortion.
    
    Args:
        frame (np.ndarray): Input frame to resize (BGR format).
        max_width (int): Maximum width for resized frame. Defaults to 1280.
        max_height (int): Maximum height for resized frame. Defaults to 720.
        maintain_aspect_ratio (bool): Whether to maintain original aspect ratio.
                                    Defaults to True.
    
    Returns:
        np.ndarray: Resized frame, or original frame if already within bounds.
    
    Example:
        >>> resized = resize_frame_to_fit(frame, max_width=1920, max_height=1080)
    """
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
    """Visualizer for drawing tracking information on video frames.
    
    This class provides methods for drawing bounding boxes, track IDs, confidence
    scores, movement trajectories, and statistics on video frames. It uses a
    consistent color scheme for each track ID to help visually distinguish
    different students.
    
    The visualizer supports customizable drawing styles and can display:
    - Bounding boxes with track-specific colors
    - Track IDs and confidence scores
    - Movement trajectories (trails showing past positions)
    - Student count overlays
    - FPS display
    
    Attributes:
        show_confidence (bool): Whether to display confidence scores
        show_track_id (bool): Whether to display track IDs
        show_trajectory (bool): Whether to draw movement trails
        box_thickness (int): Thickness of bounding box lines
        font_scale (float): Scale factor for text
        font_thickness (int): Thickness of text
        trajectory_length (int): Number of past positions to draw in trajectory
        colors (list[tuple]): Color palette for different track IDs
    
    Example:
        >>> visualizer = Visualizer(
        ...     show_confidence=True,
        ...     show_track_id=True,
        ...     show_trajectory=True,
        ...     box_thickness=2
        ... )
        >>> annotated = visualizer.draw_tracking_info(frame, tracking_results)
    """
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
        """Initialize visualizer with drawing configuration.
        
        Creates a new visualizer with specified drawing options. Each track ID
        will be assigned a unique color from the color palette.
        
        Args:
            show_confidence (bool): Display confidence scores. Defaults to True.
            show_track_id (bool): Display track IDs. Defaults to True.
            show_trajectory (bool): Draw movement trajectories. Defaults to True.
            box_thickness (int): Thickness of bounding box lines. Defaults to 2.
            font_scale (float): Scale factor for text size. Defaults to 0.6.
            font_thickness (int): Thickness of text. Defaults to 2.
            trajectory_length (int): Number of past positions in trajectory.
                                   Defaults to 50.
        
        Example:
            >>> visualizer = Visualizer(
            ...     show_confidence=True,
            ...     box_thickness=3,
            ...     font_scale=0.8
            ... )
        """
        # Store drawing configuration
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id
        self.show_trajectory = show_trajectory
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.trajectory_length = trajectory_length

        # Color palette for different track IDs (BGR format)
        self.colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 128, 255),    # Light Blue
            (255, 128, 128),  # Pink
        ]

    def _get_color(self, track_id: int) -> tuple[int, int, int]:
        """Get color for a specific track ID.
        
        Returns a consistent color for each track ID by cycling through the
        color palette. This ensures each student has a unique color throughout
        the video.
        
        Args:
            track_id (int): Track ID to get color for.
        
        Returns:
            tuple[int, int, int]: BGR color tuple.
        """
        return self.colors[track_id % len(self.colors)]

    def draw_tracking_info(
        self, frame: np.ndarray, tracking_results: dict[str, Any]
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and confidence scores on frame.
        
        This method draws visual elements for each tracked student:
        - Bounding box with track-specific color
        - Center point marker
        - Track ID label
        - Confidence score (if enabled)
        
        Args:
            frame (np.ndarray): Frame to draw on (BGR format).
            tracking_results (dict[str, Any]): Tracking results containing:
                - "boxes": List of bounding boxes
                - "confidences": List of confidence scores
                - "track_ids": List of track IDs
                - "track_positions": Dict mapping track IDs to center positions
        
        Returns:
            np.ndarray: Frame with tracking information drawn.
        
        Example:
            >>> annotated = visualizer.draw_tracking_info(frame, tracking_results)
        """
        # Extract tracking data
        boxes = tracking_results.get("boxes", [])
        confidences = tracking_results.get("confidences", [])
        track_ids = tracking_results.get("track_ids", [])
        track_positions = tracking_results.get("track_positions", {})

        # Draw each tracked student
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
        """Draw movement trajectories (trails) for tracked students.
        
        Draws lines connecting past positions of each tracked student, creating
        a visual trail showing their movement path. Older positions are drawn
        with reduced opacity for a fading effect.
        
        Args:
            frame (np.ndarray): Frame to draw on (BGR format).
            tracking_results (dict[str, Any]): Tracking results containing track_ids.
            track_histories (dict[int, list[tuple[float, float]]]): Movement history
                for each track ID as list of (x, y) positions.
        
        Returns:
            np.ndarray: Frame with trajectories drawn.
        
        Example:
            >>> annotated = visualizer.draw_trajectory(frame, tracking_results, histories)
        """
        # Skip if trajectories are disabled
        if not self.show_trajectory:
            return frame

        # Get active track IDs
        track_ids = tracking_results.get("track_ids", [])

        # Draw trajectory for each active track
        for track_id in track_ids:
            track_id = int(track_id)

            # Skip if no history for this track
            if track_id not in track_histories:
                continue

            # Get trajectory and limit length
            trajectory = track_histories[track_id]
            if len(trajectory) > self.trajectory_length:
                trajectory = trajectory[-self.trajectory_length :]

            # Get color for this track
            color = self._get_color(track_id)

            # Draw lines connecting past positions
            for i in range(1, len(trajectory)):
                pt1 = (int(trajectory[i - 1][0]), int(trajectory[i - 1][1]))
                pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))

                # Fade older positions (reduce opacity)
                alpha = i / len(trajectory)
                line_color = tuple(int(c * alpha) for c in color)

                cv2.line(frame, pt1, pt2, line_color, 2)

            # Draw small circles at each position
            for point in trajectory:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)

        return frame

    def draw_student_count(self, frame: np.ndarray, count_info: dict[str, Any]) -> np.ndarray:
        """Draw student count statistics overlay on frame.
        
        Draws a semi-transparent overlay in the top-left corner showing:
        - Current number of students present
        - Total unique students seen
        - Maximum concurrent students
        
        Args:
            frame (np.ndarray): Frame to draw on (BGR format).
            count_info (dict[str, Any]): Count statistics containing:
                - "current_count": Current number of students
                - "total_unique_students": Total unique students seen
                - "max_concurrent_students": Maximum students at once
        
        Returns:
            np.ndarray: Frame with count overlay drawn.
        
        Example:
            >>> annotated = visualizer.draw_student_count(frame, count_info)
        """
        # Extract count statistics
        current_count = count_info.get("current_count", 0)
        total_unique = count_info.get("total_unique_students", 0)
        max_concurrent = count_info.get("max_concurrent_students", 0)

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw statistics text
        y_offset = 35
        line_height = 25

        # Current count (large, green)
        count_text = f"Students Present: {current_count}"
        cv2.putText(
            frame,
            count_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),  # Green
            3,
        )

        y_offset += line_height

        # Total unique (smaller, white)
        unique_text = f"Total Unique: {total_unique}"
        cv2.putText(
            frame,
            unique_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White
            2,
        )

        y_offset += line_height

        # Max concurrent (smaller, white)
        max_text = f"Max Concurrent: {max_concurrent}"
        cv2.putText(
            frame,
            max_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White
            2,
        )

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS (frames per second) display on frame.
        
        Draws the current FPS in the top-right corner of the frame with a
        semi-transparent background for readability.
        
        Args:
            frame (np.ndarray): Frame to draw on (BGR format).
            fps (float): Current frames per second.
        
        Returns:
            np.ndarray: Frame with FPS display drawn.
        
        Example:
            >>> annotated = visualizer.draw_fps(frame, fps=30.5)
        """
        # Format FPS text
        fps_text = f"FPS: {fps:.1f}"

        # Get frame dimensions
        height, width = frame.shape[:2]
        text_x = width - 150
        text_y = 30

        # Get text size for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (text_x - 10, text_y - text_height - 10),
            (text_x + text_width + 10, text_y + 10),
            (0, 0, 0),  # Black
            -1,
        )

        # Draw FPS text
        cv2.putText(
            frame,
            fps_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Green
            2,
        )

        return frame
