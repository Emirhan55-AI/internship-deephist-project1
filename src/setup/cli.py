from __future__ import annotations

import argparse
from typing import Any, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
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
    parser = build_parser()
    args = parser.parse_args(argv)

    input_candidate: str | int | None = args.input
    if input_candidate is not None:
        try:
            input_candidate = int(input_candidate)
        except (TypeError, ValueError):
            pass

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

