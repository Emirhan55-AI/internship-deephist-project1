import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class VideoProcessor:
    def __init__(
        self,
        input_source: str | int = 0,
        output_path: str | None = None,
        output_fps: float | None = None,
        output_codec: str = "mp4v",
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_source = input_source
        self.output_path = output_path
        self.output_fps = output_fps
        self.output_codec = output_codec
        self._output_enabled = output_path is not None

        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None

        self.width: int | None = None
        self.height: int | None = None
        self.fps: float | None = None
        self.frame_count = 0
        self.total_frames = 0

        self.logger = logger or logging.getLogger(__name__)

        self._initialize_capture()

        self.logger.info(f"Video processor initialized with source: {input_source}")
        if output_path:
            self.logger.info(f"Output video will be saved to: {output_path}")

    def _initialize_capture(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.input_source)

            if not self.cap.isOpened():
                raise ValueError(
                    f"Could not open video source: {self.input_source}. "
                    f"Check if the source is valid and accessible."
                )

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if self.output_fps is None:
                self.output_fps = self.fps if self.fps > 0 else 30.0

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
        if self.cap is None:
            return False, None

        success, frame = self.cap.read()

        if success:
            self.frame_count += 1
            if self.writer is None and self.output_path is not None:
                self._initialize_writer()

        return success, frame

    def write_frame(self, frame: np.ndarray) -> bool:
        if not self.output_enabled or self.writer is None:
            return False

        try:
            frame_height, frame_width = frame.shape[:2]
            if frame_width != self.width or frame_height != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.writer.write(frame)
            return True

        except Exception as exc:
            self.logger.error(f"Failed to write frame: {exc}")
            return False

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.logger.info("Video processor resources released")

    @property
    def output_enabled(self) -> bool:
        return self._output_enabled
