"""
Video Controller Module

This module contains the VideoController class, which manages the main video
processing loop. It handles frame reading, processing, display, output writing,
keyboard controls, and statistics reporting.

The controller orchestrates the entire application lifecycle from startup to
shutdown, including:
- Video frame reading and processing
- Real-time display with keyboard controls
- Output video writing
- Progress reporting and statistics
- Graceful cleanup on exit

Example:
    >>> controller = VideoController(pipeline, video_processor, counter, ...)
    >>> controller.start()  # Begin processing loop
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import cv2


class VideoController:
    """Main video processing controller that manages the application loop.
    
    This class handles the entire video processing workflow:
    - Reads frames from video source
    - Processes frames through the pipeline
    - Displays frames in real-time
    - Writes output videos
    - Handles keyboard controls (quit, pause, reset)
    - Reports progress and statistics
    
    The controller also manages frame skipping for performance optimization
    and provides graceful shutdown with resource cleanup.
    
    Attributes:
        pipeline: Frame processing pipeline instance
        video_processor: Video I/O handler
        counter: Student counter for statistics
        tracker: Multi-object tracker
        visualizer: Visualizer for drawing
        config (dict): Application configuration
        options (dict): CLI options
        logger (logging.Logger): Logger instance
        frame_index (int): Current frame number
        current_students (int): Current number of students
        start_time (float): Application start timestamp
        frame_skip (int): Number of frames to skip between processing
        paused (bool): Whether processing is paused
        _progress_printed (bool): Whether progress has been printed
    
    Example:
        >>> controller = VideoController(
        ...     pipeline=pipeline,
        ...     video_processor=video_processor,
        ...     counter=counter,
        ...     tracker=tracker,
        ...     visualizer=visualizer,
        ...     config=config,
        ...     options=options,
        ...     logger=logger
        ... )
        >>> controller.start()  # Begin processing
    """
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
        """Start the main video processing loop.
        
        This is the main entry point that begins the video processing workflow.
        It runs until the video ends, user quits, or an error occurs.
        
        The loop performs the following operations for each frame:
        1. Check if processing should stop (duration limit reached)
        2. Read next frame from video source
        3. Process frame through pipeline (with frame skipping support)
        4. Draw FPS on frame
        5. Write frame to output video (if enabled)
        6. Display frame in window (if enabled)
        7. Report progress (if enabled)
        8. Handle keyboard input (quit, pause, reset)
        
        Keyboard Controls:
            Q or Esc: Quit the application
            P: Pause/resume processing
            R: Reset tracker and counter
        
        Raises:
            RuntimeError: If components were not properly initialized.
        
        Returns:
            None
        """
        # Validate that components were initialized
        if not self.video_processor or not self.pipeline:
            raise RuntimeError("Application components were not initialized")

        # Log startup messages
        self.logger.info("Starting Eyes on You application")
        self.logger.info("Controls: Q-Quit, P-Pause/Resume, R-Reset counter")

        # Verify video source is accessible
        if not self.video_processor.is_opened():
            self.logger.error("Could not open video source")
            return

        # Record start time for FPS calculation and duration tracking
        self.start_time = time.time()

        try:
            # Main processing loop - runs until video ends or user quits
            while True:
                # Check if duration limit has been reached
                elapsed_before_read = time.time() - self.start_time
                if self._should_stop(elapsed_before_read):
                    break

                # Read next frame from video source
                success, frame = self.video_processor.read_frame()
                if not success:
                    self.logger.info("Video stream ended or frame capture failed")
                    break

                # Process frame (with frame skipping for performance)
                # Every Nth frame is fully processed, others use cached results
                if self.frame_index % self.frame_skip == 0:
                    # Full processing: detection + tracking + counting
                    result = self.pipeline.process(frame)
                    self.current_students = int(result["counts"].get("current_count", 0))
                else:
                    # Skip mode: only visualization using cached results
                    result = self.pipeline.process_skip(frame)
                    self.current_students = int(result["counts"].get("current_count", 0))

                self.frame_index += 1

                # Calculate and draw FPS on frame
                elapsed = time.time() - self.start_time
                fps_value = self._compute_fps(elapsed)
                if fps_value is not None:
                    self.visualizer.draw_fps(result["frame"], fps_value)

                # Write frame to output video file (if enabled)
                self._write_output(result["frame"])
                
                # Display frame in window (if enabled and not paused)
                self._display_frame(result["frame"])

                # Report progress to console (if enabled)
                self._report_progress(elapsed, fps_value or 0.0)

                # Handle keyboard input
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
            # Always cleanup resources, even if interrupted
            self._finalize()

    def _compute_fps(self, elapsed: float) -> float | None:
        """Calculate current frames per second.
        
        Computes the average FPS by dividing the number of frames processed
        by the elapsed time. Returns None if elapsed time is 0 or negative.
        
        Args:
            elapsed (float): Elapsed time in seconds since start.
        
        Returns:
            float | None: Current FPS, or None if elapsed <= 0.
        """
        if elapsed <= 0:
            return None
        return self.frame_index / elapsed

    def _write_output(self, frame) -> None:
        """Write frame to output video file if output is enabled.
        
        This method checks if video output is enabled and writes the current
        frame to the output video file. Does nothing if output is disabled.
        
        Args:
            frame: Frame to write to output video.
        
        Returns:
            None
        """
        if not self.video_processor or not self.video_processor.output_enabled:
            return
        self.video_processor.write_frame(frame)

    def _display_frame(self, frame) -> None:
        """Display frame in window if display is enabled and not paused.
        
        Shows the current frame in an OpenCV window. The frame is only displayed
        if display is enabled in configuration and processing is not paused.
        
        Args:
            frame: Frame to display in window.
        
        Returns:
            None
        """
        display_cfg = self.config.get("display", {})
        if not display_cfg.get("enabled", True):
            return
        if self.paused:
            return
        window_name = display_cfg.get("window_name", "Eyes on You - Student Tracking")
        cv2.imshow(window_name, frame)

    def _should_stop(self, elapsed: float) -> bool:
        """Check if processing should stop based on duration limit.
        
        Determines whether the application should stop processing based on
        the configured maximum duration limit. Logs a message if limit is reached.
        
        Args:
            elapsed (float): Elapsed time in seconds since start.
        
        Returns:
            bool: True if duration limit reached, False otherwise.
        """
        duration_limit = self._get_duration_limit()
        if duration_limit and elapsed >= duration_limit:
            self.logger.info(f"Duration limit of {duration_limit} seconds reached")
            return True
        return False

    def _get_duration_limit(self) -> float | None:
        """Get the maximum duration limit from config or CLI options.
        
        Checks both CLI options and configuration file for the duration limit.
        CLI options take precedence over config file settings.
        
        Returns:
            float | None: Maximum duration in seconds, or None if no limit set.
        """
        performance = self.config.get("performance", {})
        return (
            self.options.get("duration")
            if self.options.get("duration") is not None
            else performance.get("max_duration")
        )

    def _poll_keyboard(self) -> str | None:
        """Poll keyboard for user input and return action.
        
        Checks for keyboard input using OpenCV's waitKey. Returns the action
        corresponding to the pressed key, or None if no relevant key was pressed.
        
        Supported keys:
            Q or Esc: Quit application
            P: Toggle pause/resume
            R: Reset tracker and counter
        
        Returns:
            str | None: Action string ("quit", "toggle_pause", "reset") or None.
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord("q"), 27):  # 'q' or Esc
            return "quit"
        if key == ord("p"):  # 'p' for pause
            return "toggle_pause"
        if key == ord("r"):  # 'r' for reset
            return "reset"
        
        return None

    def _toggle_pause(self) -> None:
        """Toggle pause state of video processing.
        
        Switches between paused and resumed states. When paused, frames are
        still processed but not displayed. Logs the current state.
        
        Returns:
            None
        """
        self.paused = not self.paused
        self.logger.info("Paused" if self.paused else "Resumed")

    def _report_progress(self, elapsed: float, fps_value: float) -> None:
        """Report processing progress to console.
        
        Displays a progress message with elapsed time, total time, FPS, and
        current student count. Only displays if progress reporting is enabled
        and a duration limit is set.
        
        Args:
            elapsed (float): Elapsed time in seconds.
            fps_value (float): Current FPS value.
        
        Returns:
            None
        """
        stats_cfg = self.config.get("statistics", {})
        if not stats_cfg.get("show_progress", True):
            return
        
        duration_limit = self._get_duration_limit()
        if not duration_limit:
            return

        # Calculate progress percentage
        progress = (elapsed / duration_limit) * 100 if duration_limit else 0.0
        
        # Format progress message
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
        
        # Print progress on same line (overwrite previous progress)
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()
        self._progress_printed = True

    def _reset_tracking(self) -> None:
        """Reset tracker and counter to initial state.
        
        Clears all tracking data and resets student counts. Useful for
        restarting tracking during a session without stopping the application.
        
        Returns:
            None
        """
        if self.counter:
            self.counter.reset()
        if self.tracker:
            self.tracker.reset()
        self.logger.info("Counter and tracker reset")

    def _finalize(self) -> None:
        """Clean up resources and display final statistics.
        
        This method is called when the application is shutting down. It:
        1. Closes progress output
        2. Releases video processor resources
        3. Destroys OpenCV windows
        4. Displays final statistics (runtime, FPS, student counts)
        
        This method is always called, even if the application is interrupted.
        
        Returns:
            None
        """
        # Close progress output with newline
        if self._progress_printed:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.logger.info("Cleaning up resources...")
        
        # Release video capture and writer
        if self.video_processor:
            self.video_processor.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Display final statistics if we have timing data
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_fps = self._compute_fps(total_time) or 0.0

            stats = self.counter.get_statistics() if self.counter else {}
            
            # Print summary statistics
            print("\n=== Final Statistics ===")
            print(f"Total runtime: {total_time:.1f} seconds")
            print(f"Total frames processed: {self.frame_index}")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Final student count: {self.current_students}")
            print(f"Total unique students: {stats.get('total_unique_students', 0)}")
            print(f"Max concurrent students: {stats.get('max_concurrent_students', 0)}")

        self.logger.info("Cleanup completed")
