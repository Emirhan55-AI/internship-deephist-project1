from __future__ import annotations

import logging
import sys
import time
from typing import Any

import cv2


class VideoController:
    def __init__(
        self,
        pipeline,
        video_processor,
        counter,
        tracker,
        visualizer,
        config: dict[str, Any],
        options: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.pipeline = pipeline
        self.video_processor = video_processor
        self.counter = counter
        self.tracker = tracker
        self.visualizer = visualizer
        self.config = config
        self.options = options
        self.logger = logger

        self.frame_index = 0
        self.current_students = 0
        self.start_time: float | None = None
        self.frame_skip = config.get("performance", {}).get("frame_skip", 1)
        self.paused = False
        self._progress_printed = False

    def start(self) -> None:
        if not self.video_processor or not self.pipeline:
            raise RuntimeError("Application components were not initialized")

        self.logger.info("Starting Eyes on You application")
        self.logger.info("Controls: Q-Quit, P-Pause/Resume, R-Reset counter")

        if not self.video_processor.is_opened():
            self.logger.error("Could not open video source")
            return

        self.start_time = time.time()

        try:
            while True:
                elapsed_before_read = time.time() - self.start_time
                if self._should_stop(elapsed_before_read):
                    break

                success, frame = self.video_processor.read_frame()
                if not success:
                    self.logger.info("Video stream ended or frame capture failed")
                    break

                if self.frame_index % self.frame_skip == 0:
                    result = self.pipeline.process(frame)
                    self.current_students = int(result["counts"].get("current_count", 0))
                else:
                    result = self.pipeline.process_skip(frame)
                    self.current_students = int(result["counts"].get("current_count", 0))

                self.frame_index += 1

                elapsed = time.time() - self.start_time
                fps_value = self._compute_fps(elapsed)
                if fps_value is not None:
                    self.visualizer.draw_fps(result["frame"], fps_value)

                self._write_output(result["frame"])
                self._display_frame(result["frame"])

                self._report_progress(elapsed, fps_value or 0.0)

                action = self._poll_keyboard()
                if action == "quit":
                    break
                if action == "toggle_pause":
                    self._toggle_pause()
                elif action == "reset":
                    self._reset_tracking()

        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as exc:
            self.logger.error(f"Unexpected error: {exc}")
            raise
        finally:
            self._finalize()

    def _compute_fps(self, elapsed: float) -> float | None:
        if elapsed <= 0:
            return None
        return self.frame_index / elapsed

    def _write_output(self, frame) -> None:
        if not self.video_processor or not self.video_processor.output_enabled:
            return
        self.video_processor.write_frame(frame)

    def _display_frame(self, frame) -> None:
        display_cfg = self.config.get("display", {})
        if not display_cfg.get("enabled", True):
            return
        if self.paused:
            return
        window_name = display_cfg.get("window_name", "Eyes on You - Student Tracking")
        cv2.imshow(window_name, frame)

    def _should_stop(self, elapsed: float) -> bool:
        duration_limit = self._get_duration_limit()
        if duration_limit and elapsed >= duration_limit:
            self.logger.info(f"Duration limit of {duration_limit} seconds reached")
            return True
        return False

    def _get_duration_limit(self) -> float | None:
        performance = self.config.get("performance", {})
        return (
            self.options.get("duration")
            if self.options.get("duration") is not None
            else performance.get("max_duration")
        )

    def _poll_keyboard(self) -> str | None:
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("p"):
            return "toggle_pause"
        if key == ord("r"):
            return "reset"
        
        return None

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.logger.info("Paused" if self.paused else "Resumed")

    def _report_progress(self, elapsed: float, fps_value: float) -> None:
        stats_cfg = self.config.get("statistics", {})
        if not stats_cfg.get("show_progress", True):
            return
        
        duration_limit = self._get_duration_limit()
        if not duration_limit:
            return

        progress = (elapsed / duration_limit) * 100 if duration_limit else 0.0
        progress_format = stats_cfg.get(
            "progress_format",
            "Progress: {progress:.1f}% ({elapsed:.1f}s/{total:.1f}s) - FPS: {fps:.1f} - Students: {students}",
        )
        message = progress_format.format(
            progress=progress,
            elapsed=elapsed,
            total=duration_limit or 0.0,
            fps=fps_value,
            students=self.current_students,
        )
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()
        self._progress_printed = True

    def _reset_tracking(self) -> None:
        """Reset counter and tracker."""
        if self.counter:
            self.counter.reset()
        if self.tracker:
            self.tracker.reset()
        self.logger.info("Counter and tracker reset")

    def _finalize(self) -> None:
        if self._progress_printed:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.logger.info("Cleaning up resources...")
        
        # Cleanup video processor
        if self.video_processor:
            self.video_processor.release()
        
        cv2.destroyAllWindows()

        if self.start_time:
            total_time = time.time() - self.start_time
            avg_fps = self._compute_fps(total_time) or 0.0

            stats = self.counter.get_statistics() if self.counter else {}
            
            print("\n=== Final Statistics ===")
            print(f"Total runtime: {total_time:.1f} seconds")
            print(f"Total frames processed: {self.frame_index}")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Final student count: {self.current_students}")
            print(f"Total unique students: {stats.get('total_unique_students', 0)}")
            print(f"Max concurrent students: {stats.get('max_concurrent_students', 0)}")

        self.logger.info("Cleanup completed")
