"""
Video I/O Processor Module

This module provides the VideoProcessor class for handling video input and output
operations. It supports reading from webcams, video files, and URLs, as well as
writing processed frames to output video files.

The processor handles codec negotiation, frame resizing, and resource management
for efficient video processing.

Example:
    >>> processor = VideoProcessor(
    ...     input_source=0,
    ...     output_path="output.mp4",
    ...     output_fps=30
    ... )
    >>> success, frame = processor.read_frame()
"""

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class VideoProcessor:
    """Video input/output processor for reading and writing video frames.
    
    This class handles all video I/O operations including:
    - Reading frames from webcams, video files, or URLs
    - Writing frames to output video files
    - Managing video codecs and formats
    - Resizing frames for output consistency
    
    The processor automatically detects video properties (width, height, FPS)
    and handles codec negotiation for reliable output video creation.
    
    Attributes:
        input_source (str | int): Video input (webcam index, file path, or URL)
        output_path (str | None): Output video file path
        output_fps (float | None): Output video frame rate
        output_codec (str): Video codec for output (e.g., "mp4v", "XVID")
        cap (cv2.VideoCapture | None): Video capture object
        writer (cv2.VideoWriter | None): Video writer object
        width (int | None): Video frame width
        height (int | None): Video frame height
        fps (float | None): Video frame rate
        frame_count (int): Number of frames read
        total_frames (int): Total frames in video
        logger (logging.Logger): Logger instance
    
    Example:
        >>> processor = VideoProcessor(
        ...     input_source="input.mp4",
        ...     output_path="output.mp4",
        ...     output_fps=30
        ... )
        >>> success, frame = processor.read_frame()
        >>> processor.write_frame(frame)
    """
    def __init__(
        self,
        input_source: str | int = 0,
        output_path: str | None = None,
        output_fps: float | None = None,
        output_codec: str = "mp4v",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize video processor with input and output configuration.
        
        Creates a new video processor and initializes video capture from the
        specified source. If output path is provided, output video writer will
        be initialized on first frame write.
        
        Args:
            input_source (str | int): Video input source:
                - int: Webcam index (0, 1, 2...)
                - str: Video file path or URL
                Defaults to 0 (first webcam).
            output_path (str | None): Output video file path. If None, no output.
                                    Defaults to None.
            output_fps (float | None): Output video frame rate. If None, uses
                                     input video FPS. Defaults to None.
            output_codec (str): Video codec for output ("mp4v", "XVID", "MJPG", etc.).
                              Defaults to "mp4v".
            logger (logging.Logger | None): Logger instance. Defaults to None.
        
        Raises:
            ValueError: If video source cannot be opened.
        
        Example:
            >>> processor = VideoProcessor(
            ...     input_source=0,
            ...     output_path="output.mp4",
            ...     output_fps=30
            ... )
        """
        # Store configuration
        self.input_source = input_source
        self.output_path = output_path
        self.output_fps = output_fps
        self.output_codec = output_codec
        self._output_enabled = output_path is not None

        # Initialize video objects (will be set by _initialize_capture)
        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None

        # Video properties (will be set by _initialize_capture)
        self.width: int | None = None
        self.height: int | None = None
        self.fps: float | None = None
        self.frame_count = 0
        self.total_frames = 0

        # Setup logger
        self.logger = logger or logging.getLogger(__name__)

        # Initialize video capture
        self._initialize_capture()

        # Log initialization
        self.logger.info(f"Video processor initialized with source: {input_source}")
        if output_path:
            self.logger.info(f"Output video will be saved to: {output_path}")

    def _initialize_capture(self) -> None:
        """Initialize video capture from input source.
        
        Opens the video source and extracts video properties (width, height, FPS).
        Sets output FPS to match input FPS if not specified.
        
        Raises:
            ValueError: If video source cannot be opened.
        
        Returns:
            None
        """
        try:
            # Open video capture
            self.cap = cv2.VideoCapture(self.input_source)

            # Verify capture opened successfully
            if not self.cap.isOpened():
                raise ValueError(
                    f"Could not open video source: {self.input_source}. "
                    f"Check if the source is valid and accessible."
                )

            # Extract video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Set output FPS if not specified
            if self.output_fps is None:
                self.output_fps = self.fps if self.fps > 0 else 30.0

            # Log video properties
            self.logger.info(
                f"Video properties: {self.width}x{self.height} @ {self.fps:.2f} FPS"
            )
            self.logger.info(
                f"Total frames: {self.total_frames if self.total_frames > 0 else 'Unknown'}"
            )

        except Exception as exc:
            self.logger.error(f"Failed to initialize video capture: {exc}")
            raise

    def _initialize_writer(self) -> None:
        if not self.output_enabled:
            return

        try:
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            codecs_to_try = [self.output_codec, "XVID", "MJPG", "X264"]
            
            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    self.writer = cv2.VideoWriter(
                        self.output_path,
                        fourcc,
                        self.output_fps,
                        (self.width, self.height),
                    )

                    if self.writer.isOpened():
                        self.logger.info(f"Video writer initialized with codec '{codec}': {self.output_path}")
                        return
                    else:
                        self.logger.warning(f"Codec '{codec}' failed, trying next...")
                        if self.writer:
                            self.writer.release()
                except Exception as e:
                    self.logger.warning(f"Codec '{codec}' failed: {e}, trying next...")
                    if self.writer:
                        self.writer.release()
                    continue
            
            raise ValueError(
                f"Could not initialize video writer with any codec for: {self.output_path}"
            )

        except Exception as exc:
            self.logger.error(f"Failed to initialize video writer: {exc}")
            raise

    def read_frame(self) -> Tuple[bool, np.ndarray | None]:
        """Read next frame from video source.
        
        Reads a single frame from the input video source. If output is enabled,
        the video writer is initialized on the first successful frame read.
        
        Returns:
            Tuple[bool, np.ndarray | None]: (success, frame)
                - success: True if frame read successfully, False otherwise
                - frame: Frame as numpy array (BGR format), or None if failed
        
        Example:
            >>> success, frame = processor.read_frame()
            >>> if success:
            ...     cv2.imshow("Frame", frame)
        """
        if self.cap is None:
            return False, None

        # Read frame from video capture
        success, frame = self.cap.read()

        if success:
            self.frame_count += 1
            # Initialize writer on first frame if output is enabled
            if self.writer is None and self.output_path is not None:
                self._initialize_writer()

        return success, frame

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write frame to output video file.
        
        Writes a frame to the output video file. The frame is automatically
        resized to match the output video dimensions if needed.
        
        Args:
            frame (np.ndarray): Frame to write (BGR format).
        
        Returns:
            bool: True if frame written successfully, False otherwise.
        
        Example:
            >>> success = processor.write_frame(frame)
            >>> if not success:
            ...     print("Failed to write frame")
        """
        if not self.output_enabled or self.writer is None:
            return False

        try:
            # Resize frame if dimensions don't match output video
            frame_height, frame_width = frame.shape[:2]
            if frame_width != self.width or frame_height != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Write frame to output video
            self.writer.write(frame)
            return True

        except Exception as exc:
            self.logger.error(f"Failed to write frame: {exc}")
            return False

    def is_opened(self) -> bool:
        """Check if video capture is open and ready.
        
        Returns:
            bool: True if video capture is open, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release all video resources.
        
        Closes video capture and writer, releasing all resources. Should be
        called when done processing video to free memory and file handles.
        
        Returns:
            None
        
        Example:
            >>> processor.release()
            >>> # All resources are now freed
        """
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Release video writer
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.logger.info("Video processor resources released")

    @property
    def output_enabled(self) -> bool:
        """Check if video output is enabled.
        
        Returns:
            bool: True if output is enabled, False otherwise.
        """
        return self._output_enabled
