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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    options = parse_cli_args()
    
    setup_logging(options["verbose"])

    try:
        config = load_config(options["config_path"])
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {exc}") from exc

    config = apply_cli_overrides(config, options)

    logger = logging.getLogger("eyes_on_you")
    
    # Setup components
    pipeline, video_processor, counter, tracker = setup_components(config, logger)
    
    # Initialize video controller
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
