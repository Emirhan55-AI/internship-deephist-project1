"""
Component Initialization Module

This module is responsible for initializing all application components based on
configuration settings. It creates and configures the detector, tracker, counter,
visualizer, video processor, and pipeline.

The module reads configuration from the YAML file and creates instances of each
component with the appropriate settings. This centralizes component creation
and makes it easy to modify initialization logic.

Example:
    >>> config = load_config("config.yaml")
    >>> logger = logging.getLogger("app")
    >>> pipeline, video, counter, tracker = setup_components(config, logger)
"""

from __future__ import annotations

import logging
from typing import Any

from src.app.pipeline import FrameProcessor
from src.core.counter import StudentCounter
from src.core.detector import YOLODetector
from src.core.tracker import BotSortTracker
from src.utils.video import VideoProcessor
from src.utils.visualization import Visualizer


def setup_components(config: dict[str, Any], logger: logging.Logger) -> tuple[FrameProcessor, VideoProcessor, StudentCounter, BotSortTracker]:
    """Initialize all application components with configuration.
    
    This function creates and configures all components needed for the application:
    - YOLODetector: For person detection in video frames
    - BotSortTracker: For multi-object tracking with ReID
    - StudentCounter: For counting and statistics
    - Visualizer: For drawing bounding boxes, tracks, and statistics
    - VideoProcessor: For reading/writing video files
    - FrameProcessor: The main pipeline that orchestrates everything
    
    Each component is initialized with settings from the configuration dictionary,
    allowing for easy customization without code changes.
    
    Args:
        config (dict[str, Any]): Configuration dictionary from load_config().
        logger (logging.Logger): Logger instance for component initialization messages.
    
    Returns:
        tuple[FrameProcessor, VideoProcessor, StudentCounter, BotSortTracker]: 
            A tuple containing:
            - pipeline: Main frame processing pipeline
            - video_processor: Video I/O handler
            - counter: Student counting and statistics
            - tracker: Multi-object tracker
    
    Example:
        >>> config = load_config("config.yaml")
        >>> logger = logging.getLogger("app")
        >>> pipeline, video, counter, tracker = setup_components(config, logger)
        >>> # Now you can use these components to process video
    """
    
    # 1. Initialize YOLO Detector for Person Detection
    
    model_cfg = config.get("model", {})
    detector = YOLODetector(
        model_path=model_cfg.get("path", "models/yolo11s.pt"),
        confidence_threshold=model_cfg.get("confidence_threshold", 0.4),
        iou_threshold=model_cfg.get("iou_threshold", 0.3),
        device=model_cfg.get("device", "cpu"),
        classes=model_cfg.get("classes", [0]),  # Class 0 = person in COCO
    )
    
    
    # 2. Initialize BoT-SORT Tracker for Multi-Object Tracking
    
    tracking_cfg = config.get("tracking", {})
    video_cfg = config.get("video", {})
    
    # Get trajectory history length for drawing movement paths
    history_length = tracking_cfg.get(
        "history_length", tracking_cfg.get("trajectory_length", 50)
    )
    
    # Get frame rate for tracking algorithm (critical for motion prediction)
    frame_rate = video_cfg.get("frame_rate", 30)
    if frame_rate is None:
        frame_rate = 30  # Default fallback if not specified
    
    tracker = BotSortTracker(
        track_high_thresh=tracking_cfg.get("track_high_thresh", 0.6),
        track_low_thresh=tracking_cfg.get("track_low_thresh", 0.1),
        new_track_thresh=tracking_cfg.get("new_track_thresh", 0.6),
        track_buffer=tracking_cfg.get("track_buffer", 30),
        match_thresh=tracking_cfg.get("match_thresh", 0.8),
        proximity_thresh=tracking_cfg.get("proximity_thresh", 0.5),
        appearance_thresh=tracking_cfg.get("appearance_thresh", 0.25),
        cmc_method=tracking_cfg.get("cmc_method", "ecc"),  # Camera motion compensation
        frame_rate=frame_rate,
        device=model_cfg.get("device", "cpu"),
        history_size=history_length,
        reid_weights_path=tracking_cfg.get("reid_weights_path"),  # OSNet for ReID
        logger=logger,
    )
    
    # 3. Initialize Student Counter for Statistics
    
    counter_cfg = config.get("counter", {})
    counter = StudentCounter(
        confidence_threshold=counter_cfg.get("confidence_threshold", 0.5),
        max_confirmed_students=counter_cfg.get("max_confirmed_students", 1000),
    )
    
    
    # 4. Initialize Visualizer for Drawing on Frames
    
    visualization_cfg = config.get("visualization", {})
    visualizer = Visualizer(
        show_confidence=visualization_cfg.get("show_confidence", True),
        show_track_id=visualization_cfg.get("show_track_id", True),
        show_trajectory=visualization_cfg.get("show_trajectory", True),
        box_thickness=visualization_cfg.get("box_thickness", 2),
        font_scale=visualization_cfg.get("font_scale", 0.6),
        font_thickness=visualization_cfg.get("font_thickness", 2),
        trajectory_length=tracking_cfg.get("trajectory_length", history_length),
    )
    
    
    # 5. Initialize Video Processor for I/O Operations
    
    video_cfg = config.get("video", {})
    output_cfg = video_cfg.get("output", {})
    
    # Only set output path if output is enabled
    output_path = output_cfg.get("path") if output_cfg.get("enabled") else None
    
    video_processor = VideoProcessor(
        input_source=video_cfg.get("input_source", 0),
        output_path=output_path,
        output_fps=output_cfg.get("fps"),
        output_codec=output_cfg.get("codec", "mp4v"),
        logger=logger,
    )
    
    
    # 6. Initialize Main Pipeline (Orchestrates All Components)
    
    display_cfg = config.get("display", {})
    pipeline = FrameProcessor(
        detector=detector,
        tracker=tracker,
        counter=counter,
        visualizer=visualizer,
        resize_config=display_cfg.get("resize", {}),
        visualization_config=visualization_cfg,
    )
    
    logger.info("All components initialized successfully")
    
    return pipeline, video_processor, counter, tracker

