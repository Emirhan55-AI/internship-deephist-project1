"""
BoT-SORT Tracker Module

This module provides a wrapper around the BoT-SORT (Boosting Tracking and Sorting)
multi-object tracking algorithm. BoT-SORT combines motion and appearance features
for robust person tracking across video frames.

BoT-SORT features:
- Kalman filter for motion prediction
- OSNet for appearance-based re-identification
- Camera motion compensation (CMC) for handling camera movement
- Track buffer for handling occlusions

Example:
    >>> tracker = BotSortTracker(
    ...     track_high_thresh=0.6,
    ...     device="cpu",
    ...     reid_weights_path="models/osnet_x0_25_msmt17.pt"
    ... )
    >>> tracking_results = tracker.update(detections, frame)
"""

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
    """BoT-SORT multi-object tracker wrapper.
    
    This class provides a wrapper around the BoT-SORT tracking algorithm for
    robust multi-object tracking. BoT-SORT combines:
    - Motion features (Kalman filter for prediction)
    - Appearance features (OSNet for re-identification)
    - Camera motion compensation (CMC for camera movement)
    
    The tracker maintains track IDs across frames and provides movement history
    for each tracked object, enabling trajectory visualization.
    
    Attributes:
        track_high_thresh (float): High confidence threshold for tracks (0.0-1.0)
        track_low_thresh (float): Low confidence threshold for tracks (0.0-1.0)
        new_track_thresh (float): Threshold for creating new tracks (0.0-1.0)
        track_buffer (int): Frames to keep tracks after they disappear
        match_thresh (float): Threshold for matching tracks (0.0-1.0)
        proximity_thresh (float): Proximity threshold for matching (0.0-1.0)
        appearance_thresh (float): Appearance similarity threshold (0.0-1.0)
        cmc_method (str): Camera motion compensation method ("ecc" or "sparse")
        frame_rate (int): Video frame rate for motion prediction
        device (str): Processing device ("cpu" or "cuda")
        history_size (int): Number of past positions to store per track
        reid_weights_path (Path | None): Path to OSNet ReID model weights
        tracker (BotSort): Underlying BoT-SORT tracker instance
        track_history (dict[int, deque]): Movement history for each track ID
    
    Example:
        >>> tracker = BotSortTracker(
        ...     track_high_thresh=0.6,
        ...     track_low_thresh=0.1,
        ...     device="cpu",
        ...     reid_weights_path="models/osnet_x0_25_msmt17.pt",
        ...     frame_rate=30
        ... )
        >>> results = tracker.update(detections, frame)
        >>> print(f"Tracking {len(results['active_ids'])} students")
    """
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
        """Initialize BoT-SORT tracker with configuration.
        
        Creates a new tracker instance with specified parameters and loads the
        OSNet ReID model for appearance-based tracking.
        
        Args:
            track_high_thresh (float): High confidence threshold for tracks (0.0-1.0).
                                     Defaults to 0.6.
            track_low_thresh (float): Low confidence threshold for tracks (0.0-1.0).
                                    Defaults to 0.1.
            new_track_thresh (float): Threshold for creating new tracks (0.0-1.0).
                                    Defaults to 0.6.
            track_buffer (int): Frames to keep tracks after they disappear.
                              Defaults to 30.
            match_thresh (float): Threshold for matching tracks (0.0-1.0).
                                Defaults to 0.8.
            proximity_thresh (float): Proximity threshold for matching (0.0-1.0).
                                    Defaults to 0.5.
            appearance_thresh (float): Appearance similarity threshold (0.0-1.0).
                                     Defaults to 0.25.
            cmc_method (str): Camera motion compensation method ("ecc" or "sparse").
                            Defaults to "ecc".
            frame_rate (int): Video frame rate for motion prediction.
                            Defaults to 30.
            device (str): Processing device ("cpu" or "cuda").
                        Defaults to "cpu".
            history_size (int): Number of past positions to store per track.
                              Defaults to 50.
            reid_weights_path (str | Path | None): Path to OSNet ReID model weights.
                                                  Defaults to None (uses default).
            logger (logging.Logger | None): Logger instance. Defaults to None.
        
        Raises:
            AssertionError: If thresholds are out of valid range.
            ImportError: If boxmot package is not installed.
        
        Example:
            >>> tracker = BotSortTracker(
            ...     track_high_thresh=0.6,
            ...     device="cuda",
            ...     frame_rate=30,
            ...     reid_weights_path="models/osnet_x0_25_msmt17.pt"
            ... )
        """
        # Validate threshold ranges
        assert 0.0 <= track_high_thresh <= 1.0, "track_high_thresh must be between 0 and 1"
        assert 0.0 <= track_low_thresh <= 1.0, "track_low_thresh must be between 0 and 1"
        assert 0.0 <= new_track_thresh <= 1.0, "new_track_thresh must be between 0 and 1"

        # Store configuration
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

        # Resolve ReID weights path
        self.reid_weights_path = self._resolve_weights_path(reid_weights_path)

        # Build BoT-SORT tracker instance
        self.tracker = self._build_tracker()

        # Initialize track history storage
        self.track_history: dict[int, deque[tuple[float, float]]] = {}
        self.history_size = max(1, history_size)

    def _resolve_weights_path(
        self, reid_weights_path: str | Path | None
    ) -> Path | None:
        """Resolve and validate ReID weights path.
        
        Checks if the provided ReID weights path exists. Returns None if the
        path doesn't exist, allowing the tracker to use default weights.
        
        Args:
            reid_weights_path (str | Path | None): Path to ReID model weights.
        
        Returns:
            Path | None: Validated path if exists, None otherwise.
        """
        if not reid_weights_path:
            return None

        candidate = Path(reid_weights_path)
        return candidate if candidate.exists() else None

    def _build_tracker(self) -> BotSort:
        """Build and configure the BoT-SORT tracker instance.
        
        Creates a new BoT-SORT tracker with all configured parameters and
        loads the OSNet ReID model for appearance-based tracking.
        
        Returns:
            BotSort: Configured BoT-SORT tracker instance.
        """
        # Use provided weights or default OSNet model
        weights = self.reid_weights_path or Path("models/osnet_x0_25_msmt17.pt")

        # Create and configure tracker
        return BotSort(
            reid_weights=weights,  # OSNet for appearance features
            device=self.device,
            half=False,  # Use full precision (FP32)
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            cmc_method=self.cmc_method,  # Camera motion compensation
            frame_rate=self.frame_rate,
        )

    def update(self, detections: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
        """Update tracker with new detections.
        
        This is the main tracking method that processes detections from the
        detector and updates track IDs using BoT-SORT. It maintains track
        consistency across frames and stores movement history.
        
        The method:
        1. Extracts detection data (boxes, confidences, class IDs)
        2. Converts to BoT-SORT format
        3. Runs tracking algorithm
        4. Updates track history for trajectory visualization
        5. Returns tracking results
        
        Args:
            detections (dict[str, Any]): Detection results from detector containing:
                - "boxes": List of bounding boxes [x1, y1, x2, y2]
                - "confidences": List of confidence scores
                - "class_ids": List of class IDs
            frame (np.ndarray): Current video frame for appearance features.
        
        Returns:
            dict[str, Any]: Tracking results containing:
                - "boxes": Array of tracked bounding boxes
                - "confidences": Array of confidence scores
                - "track_ids": Array of track IDs (persistent across frames)
                - "class_ids": Array of class IDs
                - "active_ids": Set of active track IDs
                - "track_positions": Dict mapping track IDs to (x, y) centers
                - "track_confidences": Dict mapping track IDs to confidence scores
        
        Example:
            >>> results = tracker.update(detections, frame)
            >>> print(f"Tracking {len(results['active_ids'])} students")
            >>> for track_id in results['active_ids']:
            ...     x, y = results['track_positions'][track_id]
            ...     print(f"Track {track_id} at ({x:.1f}, {y:.1f})")
        """
        # Extract detection data
        boxes = detections.get("boxes", [])
        confidences = detections.get("confidences", [])
        class_ids = detections.get("class_ids", [])

        # Return empty results if no detections
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

        # Convert detections to BoT-SORT format [x1, y1, x2, y2, conf, class_id]
        dets = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            cls_id = class_ids[i]
            dets.append([x1, y1, x2, y2, conf, cls_id])

        dets = np.array(dets)

        # Run BoT-SORT tracking algorithm
        tracked_objects = self.tracker.update(dets, frame)

        # Extract tracking results
        tracked_boxes = []
        tracked_confidences = []
        tracked_ids = []
        tracked_class_ids = []
        track_positions = {}
        track_confidences = {}

        # Process each tracked object
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, conf, cls_id = obj[:7]

            # Store bounding box and confidence
            tracked_boxes.append([x1, y1, x2, y2])
            tracked_confidences.append(conf)
            
            # Convert track_id to integer
            track_id_int = int(track_id)
            tracked_ids.append(track_id_int)
            tracked_class_ids.append(int(cls_id))

            # Calculate center position for trajectory tracking
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            track_positions[track_id_int] = (center_x, center_y)
            track_confidences[track_id_int] = float(conf)

            # Update movement history for this track
            history = self.track_history.setdefault(
                track_id_int, deque(maxlen=self.history_size)
            )
            history.append((center_x, center_y))

        # Return comprehensive tracking results
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
        """Get movement history for a specific track.
        
        Returns the past positions of a tracked object, which can be used to
        draw movement trajectories.
        
        Args:
            track_id (int): Track ID to get history for.
        
        Returns:
            list[tuple[float, float]]: List of (x, y) positions in chronological order.
                                     Returns empty list if track_id not found.
        
        Example:
            >>> history = tracker.get_track_history(track_id=1)
            >>> for x, y in history:
            ...     print(f"Position: ({x:.1f}, {y:.1f})")
        """
        history = self.track_history.get(track_id)
        if history is None:
            return []
        return list(history)

    def reset(self) -> None:
        """Reset tracker to initial state.
        
        Clears all track history and rebuilds the tracker. Useful for
        restarting tracking during a session without stopping the application.
        
        Returns:
            None
        
        Example:
            >>> tracker.reset()
            >>> # All tracks are now cleared
        """
        # Clear movement history
        self.track_history.clear()
        
        # Rebuild tracker with same configuration
        self.tracker = self._build_tracker()

