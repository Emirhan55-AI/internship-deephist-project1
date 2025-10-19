"""
This module serves as the entry point for the Eyes on You application, a real-time
student detection and tracking system using YOLO11s for detection and BoT-SORT for
multi-object tracking.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.video_controller import VideoController
from src.setup import apply_cli_overrides, load_config, parse_cli_args, setup_components


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.
    
    Sets up the logging system with appropriate level and format based on the
    verbosity setting. DEBUG level provides detailed information for debugging,
    while INFO level shows only important messages.
    
    Args:
        verbose (bool): If True, sets logging level to DEBUG. Otherwise, uses INFO.
                        Defaults to False.
    
    Returns:
        None
    
    Example:
        >>> setup_logging(verbose=True)  # Enable debug logging
        >>> setup_logging(verbose=False)  # Use info logging (default)
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Main entry point for the Eyes on You application.
    
    This function orchestrates the entire application lifecycle:
    1. Parses command-line arguments
    2. Configures logging based on verbosity
    3. Loads and validates configuration from YAML file
    4. Applies CLI overrides to configuration
    5. Initializes all components (detector, tracker, counter, pipeline)
    6. Starts the video processing loop
    
    The application will process video frames, detect and track students,
    and display/record results until the user quits or the video ends.
    
    Raises:
        SystemExit: If configuration file is not found or other critical errors occur.
    
    Returns:
        None
    
    Example:
        >>> main()  # Start the application
    """
    # Parse command-line arguments
    options = parse_cli_args()
    
    # Configure logging based on verbosity flag
    setup_logging(options["verbose"])

    # Load configuration from YAML file
    try:
        config = load_config(options["config_path"])
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {exc}") from exc

    # Apply CLI argument overrides to configuration
    config = apply_cli_overrides(config, options)

    # Get logger instance for the application
    logger = logging.getLogger("eyes_on_you")
    
    # Initialize all components (detector, tracker, counter, pipeline)
    pipeline, video_processor, counter, tracker = setup_components(config, logger)
    
    # Create and start video controller to begin processing
    video_controller = VideoController(
        pipeline=pipeline,
        video_processor=video_processor,
        counter=counter,
        tracker=tracker,
        visualizer=pipeline.visualizer,
        config=config,
        options=options,
        logger=logger,
    )
    video_controller.start()


if __name__ == "__main__":
    main()
