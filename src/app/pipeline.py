"""
Frame Processing Pipeline Module

This module contains the FrameProcessor class, which orchestrates the entire
frame processing pipeline. It coordinates detection, tracking, counting, and
visualization for each video frame.

The pipeline processes frames through the following steps:
1. Resize frame if needed
2. Detect students using YOLO
3. Track students using BoT-SORT
4. Count students and update statistics
5. Draw visualizations (boxes, IDs, trajectories, counts)

Example:
    >>> processor = FrameProcessor(detector, tracker, counter, visualizer, ...)
    >>> result = processor.process(frame)
    >>> annotated_frame = result["frame"]
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.utils.visualization import Visualizer, resize_frame_to_fit


class FrameProcessor:
    """Main frame processing pipeline that orchestrates all components.
    
    This class coordinates the entire frame processing workflow:
    - Resizes frames if needed
    - Detects students using YOLO
    - Tracks students using BoT-SORT
    - Counts students and maintains statistics
    - Draws visualizations (bounding boxes, track IDs, trajectories, counts)
    
    The class also supports frame skipping for performance optimization,
    where detection and tracking are skipped on certain frames but visualization
    is still applied using cached results.
    
    Attributes:
        detector: YOLO detector instance for person detection
        tracker: BoT-SORT tracker instance for multi-object tracking
        counter: Student counter instance for statistics
        visualizer: Visualizer instance for drawing on frames
        resize_enabled (bool): Whether frame resizing is enabled
        resize_width (int): Target width for resizing
        resize_height (int): Target height for resizing
        maintain_aspect (bool): Whether to maintain aspect ratio when resizing
        show_trajectory (bool): Whether to draw movement trajectories
        last_tracking (dict): Cached tracking results from last frame
        last_counts (dict): Cached count results from last frame
    
    Example:
        >>> processor = FrameProcessor(
        ...     detector=detector,
        ...     tracker=tracker,
        ...     counter=counter,
        ...     visualizer=visualizer,
        ...     resize_config={"enabled": True, "width": 1280, "height": 720},
        ...     visualization_config={"show_trajectory": True}
        ... )
        >>> result = processor.process(frame)
        >>> annotated_frame = result["frame"]
        >>> tracking_data = result["tracking"]
        >>> student_count = result["counts"]["current_count"]
    """
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
        """Process a single video frame through the complete pipeline.
        
        This is the main method that processes a frame through all stages:
        1. Resize if needed
        2. Detect students (YOLO)
        3. Track students (BoT-SORT)
        4. Count students
        5. Draw visualizations
        
        The results are cached for use in frame skipping mode.
        
        Args:
            frame (np.ndarray): Input video frame as numpy array (BGR format).
        
        Returns:
            dict[str, Any]: Dictionary containing:
                - "frame": Annotated frame with visualizations
                - "tracking": Tracking results with bounding boxes and IDs
                - "counts": Student count statistics
        
        Raises:
            ValueError: If frame is None or invalid.
        
        Example:
            >>> result = processor.process(frame)
            >>> annotated_frame = result["frame"]
            >>> current_students = result["counts"]["current_count"]
        """
        # Validate input
        if frame is None:
            raise ValueError("FrameProcessor.process received a None frame")

        # Step 1: Resize frame if needed (for display optimization)
        working_frame = self._maybe_resize(frame)

        # Step 2: Detect students using YOLO
        detections = self.detector.detect(working_frame)
        
        # Step 3: Track students using BoT-SORT
        tracking = self.tracker.update(detections, working_frame)
        
        # Step 4: Count students and update statistics
        counts = self.counter.update_count(tracking)

        # Cache results for frame skipping mode
        self.last_tracking = tracking
        self.last_counts = counts

        # Step 5: Draw visualizations (boxes, IDs, trajectories, counts)
        annotated_frame = self._draw_visuals(working_frame, tracking, counts)

        return {
            "frame": annotated_frame,
            "tracking": tracking,
            "counts": counts,
        }
    
    def process_skip(self, frame: np.ndarray) -> dict[str, Any]:
        """Process a frame in skip mode (no detection/tracking).
        
        This method is used when frame_skip > 1 to improve performance.
        It skips detection and tracking but still draws visualizations using
        cached results from the last processed frame.
        
        This is useful for:
        - Faster processing when detection/tracking is expensive
        - Maintaining smooth visualization while reducing computation
        - Testing visualization settings without full processing
        
        Args:
            frame (np.ndarray): Input video frame as numpy array (BGR format).
        
        Returns:
            dict[str, Any]: Dictionary with annotated frame and cached results.
        
        Raises:
            ValueError: If frame is None or invalid.
        
        Example:
            >>> # When frame_skip=2, this is called for every other frame
            >>> result = processor.process_skip(frame)
            >>> # Uses cached tracking results from last process() call
        """
        # Validate input
        if frame is None:
            raise ValueError("FrameProcessor.process_skip received a None frame")

        # Resize frame for display
        working_frame = self._maybe_resize(frame)

        # Draw visualizations using cached results (no detection/tracking)
        annotated_frame = self._draw_visuals(working_frame, self.last_tracking, self.last_counts)

        return {
            "frame": annotated_frame,
            "tracking": self.last_tracking,
            "counts": self.last_counts,
        }

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if resizing is enabled.
        
        This helper method checks if resizing is enabled and applies it if needed.
        Resizing is useful for:
        - Reducing computation by processing smaller frames
        - Optimizing display performance
        - Standardizing frame sizes
        
        Args:
            frame (np.ndarray): Input frame to potentially resize.
        
        Returns:
            np.ndarray: Resized frame if enabled, otherwise original frame.
        """
        # Return original frame if resizing is disabled
        if not self.resize_enabled:
            return frame
        
        # Return original frame if dimensions are not specified
        if not self.resize_width or not self.resize_height:
            return frame
        
        # Apply resizing with aspect ratio preservation
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
        """Draw all visualizations on the frame.
        
        This method orchestrates the drawing of all visual elements:
        - Bounding boxes around detected students
        - Track IDs for each student
        - Confidence scores (if enabled)
        - Movement trajectories (if enabled)
        - Student count display
        
        Args:
            frame (np.ndarray): Frame to draw on.
            tracking_results (dict[str, Any]): Tracking results with bounding boxes and IDs.
            counts (dict[str, Any]): Student count statistics.
        
        Returns:
            np.ndarray: Frame with all visualizations drawn.
        """
        # Draw bounding boxes, track IDs, and confidence scores
        annotated = self.visualizer.draw_tracking_info(frame, tracking_results)

        # Draw movement trajectories if enabled
        if self.show_trajectory:
            histories = self._collect_histories(tracking_results)
            if histories:
                annotated = self.visualizer.draw_trajectory(annotated, tracking_results, histories)

        # Draw student count overlay
        annotated = self.visualizer.draw_student_count(annotated, counts)
        return annotated

    def _collect_histories(
        self, tracking_results: dict[str, Any]
    ) -> dict[int, Iterable[tuple[float, float]]]:
        """Collect movement histories for all active tracks.
        
        This method extracts the movement history (trajectory) for each
        currently active student track. The history contains past positions
        that can be used to draw movement trails.
        
        Args:
            tracking_results (dict[str, Any]): Tracking results containing active_ids.
        
        Returns:
            dict[int, Iterable[tuple[float, float]]]: Dictionary mapping track IDs
                to their position histories as (x, y) tuples.
        """
        # Initialize set for active track IDs
        active_ids: set[int] = set()
        raw_ids = tracking_results.get("active_ids", set())

        # Convert active_ids to a set of integers
        # Handles different input types (numpy array, list, set, tuple)
        if isinstance(raw_ids, np.ndarray):
            active_ids = {int(item) for item in raw_ids.tolist()}
        elif isinstance(raw_ids, (list, set, tuple)):
            active_ids = {int(item) for item in raw_ids}

        # Get position history for each active track
        histories: dict[int, Iterable[tuple[float, float]]] = {}
        for track_id in active_ids:
            histories[track_id] = self.tracker.get_track_history(track_id)
        return histories
