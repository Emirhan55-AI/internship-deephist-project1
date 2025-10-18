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
    """Initialize all components with configuration.
    
    Returns:
        Tuple of (pipeline, video_processor, counter, tracker)
    """
    # Initialize detector
    model_cfg = config.get("model", {})
    detector = YOLODetector(
        model_path=model_cfg.get("path", "models/yolo11s.pt"),
        confidence_threshold=model_cfg.get("confidence_threshold", 0.4),
        iou_threshold=model_cfg.get("iou_threshold", 0.3),
        device=model_cfg.get("device", "cpu"),
        classes=model_cfg.get("classes", [0]),
    )
    
    # Initialize tracker
    tracking_cfg = config.get("tracking", {})
    video_cfg = config.get("video", {})
    history_length = tracking_cfg.get(
        "history_length", tracking_cfg.get("trajectory_length", 50)
    )
    
    # Get frame_rate from video config or use default
    frame_rate = video_cfg.get("frame_rate", 30)
    if frame_rate is None:
        frame_rate = 30  # Default fallback
    
    tracker = BotSortTracker(
        track_high_thresh=tracking_cfg.get("track_high_thresh", 0.6),
        track_low_thresh=tracking_cfg.get("track_low_thresh", 0.1),
        new_track_thresh=tracking_cfg.get("new_track_thresh", 0.6),
        track_buffer=tracking_cfg.get("track_buffer", 30),
        match_thresh=tracking_cfg.get("match_thresh", 0.8),
        proximity_thresh=tracking_cfg.get("proximity_thresh", 0.5),
        appearance_thresh=tracking_cfg.get("appearance_thresh", 0.25),
        cmc_method=tracking_cfg.get("cmc_method", "ecc"),
        frame_rate=frame_rate,
        device=model_cfg.get("device", "cpu"),
        history_size=history_length,
        reid_weights_path=tracking_cfg.get("reid_weights_path"),
        logger=logger,
    )
    
    # Initialize counter
    counter_cfg = config.get("counter", {})
    counter = StudentCounter(
        confidence_threshold=counter_cfg.get("confidence_threshold", 0.5),
        max_confirmed_students=counter_cfg.get("max_confirmed_students", 1000),
    )
    
    # Initialize visualizer
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
    
    # Initialize video processor
    video_cfg = config.get("video", {})
    output_cfg = video_cfg.get("output", {})
    output_path = output_cfg.get("path") if output_cfg.get("enabled") else None
    
    video_processor = VideoProcessor(
        input_source=video_cfg.get("input_source", 0),
        output_path=output_path,
        output_fps=output_cfg.get("fps"),
        output_codec=output_cfg.get("codec", "mp4v"),
        logger=logger,
    )
    
    # Initialize pipeline
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

