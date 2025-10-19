"""
Command-Line Interface (CLI) Module

This module handles command-line argument parsing for the Eyes on You application.
It defines all available CLI options and provides user-friendly help messages.

The CLI allows users to override configuration file settings without modifying
the YAML file, making it easy to test different settings quickly.

Example:
    >>> options = parse_cli_args()
    >>> print(options["input_source"])
"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser.
    
    Creates an ArgumentParser with all available CLI options for the application.
    Each option can override corresponding settings in the configuration file.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all options.
    
    Example:
        >>> parser = build_parser()
        >>> args = parser.parse_args(["--input", "0", "--verbose"])
    """
    parser = argparse.ArgumentParser(
        description="Eyes on You - Student Detection and Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config.yaml
  python main.py --input 0 --width 1280 --height 720
  python main.py --input video.mp4 --output output.mp4 --duration 60
  python main.py --input rtsp://camera-url --no-display --verbose
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Video source (camera index, file path, or URL)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Optional output video path",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Override YOLO model path",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable on-screen display",
    )

    parser.add_argument(
        "--width",
        type=int,
        help="Resize width for display output",
    )

    parser.add_argument(
        "--height",
        type=int,
        help="Resize height for display output",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Limit runtime duration in seconds",
    )

    return parser


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Parse command-line arguments and return as a dictionary.
    
    This function processes command-line arguments and converts them into a
    structured dictionary that can be used to override configuration settings.
    
    The input_source is automatically converted to an integer if it's a valid
    number (for webcam indices), otherwise it's kept as a string (for file paths).
    
    Args:
        argv (Optional[Sequence[str]]): Command-line arguments to parse.
                                       If None, uses sys.argv. Defaults to None.
    
    Returns:
        dict[str, Any]: Dictionary containing parsed arguments with keys:
            - config_path: Path to YAML configuration file
            - input_source: Video source (int for webcam, str for file/URL)
            - output_path: Output video file path
            - model_path: Override YOLO model path
            - disable_display: Whether to disable display window
            - width: Display window width
            - height: Display window height
            - verbose: Enable verbose logging
            - duration: Maximum runtime duration in seconds
    
    Example:
        >>> options = parse_cli_args(["--input", "0", "--verbose"])
        >>> print(options["input_source"])  # 0 (converted to int)
        >>> print(options["verbose"])  # True
    """
    # Build and parse arguments
    parser = build_parser()
    args = parser.parse_args(argv)

    # Try to convert input_source to integer if it's a webcam index
    # Webcam indices are integers (0, 1, 2...), file paths remain strings
    input_candidate: str | int | None = args.input
    if input_candidate is not None:
        try:
            input_candidate = int(input_candidate)
        except (TypeError, ValueError):
            # Not a number, keep as string (file path or URL)
            pass

    # Return structured dictionary with all parsed arguments
    return {
        "config_path": args.config,
        "input_source": input_candidate,
        "output_path": args.output,
        "model_path": args.model,
        "disable_display": args.no_display,
        "width": args.width,
        "height": args.height,
        "verbose": args.verbose,
        "duration": args.duration,
    }

