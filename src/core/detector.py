"""
YOLO Detector Module

This module provides a wrapper around the Ultralytics YOLO model for person detection.
It simplifies the interface to YOLO and provides easy configuration of detection
parameters such as confidence threshold, IoU threshold, and device selection.

The detector processes video frames and returns bounding boxes, confidence scores,
and class information for detected persons.

Example:
    >>> detector = YOLODetector(
    ...     model_path="models/yolo11s.pt",
    ...     confidence_threshold=0.4,
    ...     device="cpu"
    ... )
    >>> detections = detector.detect(frame)
    >>> print(f"Found {len(detections['boxes'])} persons")
"""

from typing import Any

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """YOLO-based person detector wrapper.
    
    This class provides a simplified interface to the Ultralytics YOLO model for
    detecting persons in video frames. It handles model loading, configuration,
    and provides a clean API for detection.
    
    The detector uses YOLO11s (small variant) by default for optimal speed-accuracy
    balance in real-time applications. It can detect persons (COCO class 0) or
    other objects depending on configuration.
    
    Attributes:
        model_path (str): Path to YOLO model weights file (.pt)
        confidence_threshold (float): Minimum confidence for detections (0.0-1.0)
        iou_threshold (float): IoU threshold for NMS (0.0-1.0)
        classes (list[int]): COCO class IDs to detect (0=person)
        device (str): Processing device ("cpu" or "cuda")
        model (YOLO): Loaded YOLO model instance
    
    Example:
        >>> detector = YOLODetector(
        ...     model_path="models/yolo11s.pt",
        ...     confidence_threshold=0.4,
        ...     iou_threshold=0.5,
        ...     device="cpu",
        ...     classes=[0]  # Person class
        ... )
        >>> detections = detector.detect(frame)
        >>> boxes = detections["boxes"]  # Bounding boxes
        >>> confidences = detections["confidences"]  # Confidence scores
    """
    def __init__(
        self,
        model_path: str = "models/yolo11s.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.3,
        device: str | None = None,
        classes: list[int] | None = None,
    ) -> None:
        """Initialize YOLO detector with configuration.
        
        Creates a new detector instance and loads the YOLO model. The detector
        will be ready to process frames after initialization.
        
        Args:
            model_path (str): Path to YOLO model weights file. Defaults to "models/yolo11s.pt".
            confidence_threshold (float): Minimum confidence for detections (0.0-1.0).
                                        Higher values = fewer but more accurate detections.
                                        Defaults to 0.4.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression (0.0-1.0).
                                 Higher values = more aggressive duplicate removal.
                                 Defaults to 0.3.
            device (str | None): Processing device ("cpu" or "cuda"). If None, uses "cpu".
            classes (list[int] | None): COCO class IDs to detect. If None, detects persons (class 0).
        
        Raises:
            AssertionError: If confidence_threshold or iou_threshold are out of valid range.
            RuntimeError: If model file is not found or cannot be loaded.
        
        Example:
            >>> detector = YOLODetector(
            ...     model_path="models/yolo11s.pt",
            ...     confidence_threshold=0.4,
            ...     iou_threshold=0.5,
            ...     device="cpu"
            ... )
        """
        # Validate threshold ranges
        assert 0.0 <= confidence_threshold <= 1.0, "confidence_threshold must be between 0 and 1"
        assert 0.0 <= iou_threshold <= 1.0, "iou_threshold must be between 0 and 1"

        # Store configuration
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or [0]  # Default to person class
        self.device = device or "cpu"

        # Model will be loaded by _load_model()
        self.model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model from the specified path.
        
        This method loads the YOLO model weights into memory. The model file
        must exist and be a valid YOLO checkpoint (.pt file).
        
        Raises:
            RuntimeError: If model file is not found or cannot be loaded.
        
        Returns:
            None
        """
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
        """Detect persons in a video frame.
        
        This is the main detection method that processes a frame and returns
        all detected persons with their bounding boxes, confidence scores,
        and class information.
        
        The detection process:
        1. Runs YOLO inference on the frame
        2. Filters detections by confidence threshold
        3. Applies Non-Maximum Suppression (NMS) to remove duplicates
        4. Returns structured detection results
        
        Args:
            frame (np.ndarray): Input video frame as numpy array (BGR format).
        
        Returns:
            dict[str, Any]: Dictionary containing:
                - "boxes": List of bounding boxes as [x1, y1, x2, y2] arrays
                - "confidences": List of confidence scores (0.0-1.0)
                - "class_ids": List of COCO class IDs
                - "class_names": List of class names (e.g., "person")
                - "raw_result": Raw YOLO result object
        
        Raises:
            RuntimeError: If model is not loaded or detection fails.
            ValueError: If frame is None or empty.
        
        Example:
            >>> detections = detector.detect(frame)
            >>> for i, box in enumerate(detections["boxes"]):
            ...     x1, y1, x2, y2 = box
            ...     confidence = detections["confidences"][i]
            ...     print(f"Person at ({x1}, {y1}) with confidence {confidence:.2f}")
        """
        # Validate model is loaded
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot perform detection.")

        # Validate input frame
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided to detector")

        try:
            # Run YOLO inference on the frame
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,  # Filter by confidence
                iou=self.iou_threshold,  # NMS IoU threshold
                classes=self.classes,  # Only detect specified classes
                device=self.device,  # CPU or CUDA
                verbose=False,  # Disable YOLO's verbose output
            )

            # Extract first result (single frame)
            result = results[0]

            # Initialize detection dictionary
            detections: dict[str, Any] = {
                "boxes": [],
                "confidences": [],
                "class_ids": [],
                "class_names": [],
                "raw_result": result,
            }

            # Extract detection data if any detections were found
            if result.boxes is not None and len(result.boxes) > 0:
                # Bounding boxes in xyxy format (x1, y1, x2, y2)
                boxes = result.boxes.xyxy.cpu().numpy()
                
                # Confidence scores for each detection
                confidences = result.boxes.conf.cpu().numpy()
                
                # COCO class IDs (0=person, 2=bicycle, etc.)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Class names (e.g., "person", "bicycle")
                class_names = [result.names[cls_id] for cls_id in class_ids]

                # Store all detection data
                detections["boxes"] = boxes
                detections["confidences"] = confidences
                detections["class_ids"] = class_ids
                detections["class_names"] = class_names

            return detections

        except Exception as exc:
            raise RuntimeError(f"Detection failed: {exc}") from exc

