# Eyes on You - Student Detection and Tracking

Eyes on You is a modular computer vision application that detects, tracks, and counts students in real time. It combines Ultralytics YOLO11s for person detection with BoT-SORT for multi-object tracking, and renders a live overlay with per-student trajectories and statistics.

## Overview

- **Real-time Detection** - Uses YOLO11s for accurate, low-latency person detection.
- **Robust Tracking** - Integrates BoT-SORT with ReID embeddings to minimise ID switches.
- **Student Counting** - Maintains unique student totals and concurrent occupancy metrics.
- **Visualisation** - Draws tracks, trajectories, counts, and FPS directly on the feed.
- **Configuration File** - A central YAML file drives the entire pipeline.

## Quick Start

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application
   ```bash
   python main.py --config config.yaml # or 
   py -3.11 main.py --config config.yaml # if you are using python 3.13 version.
   ```

By default the application reads from `data/input/classroom_1.mp4`. Pass `--input 0` for a webcam or provide a different path/URL.

## Runtime Controls

- `Q` or `Esc` - Quit
- `P` - Pause or resume the on-screen display
- `R` - Reset active tracks and counters

## Configuration

All runtime options live in `config.yaml`. Key sections:

> **For detailed parameter explanations and tuning tips, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md)**
> **For algorithm selection rationale and literature review, see [RESEARCH.md](RESEARCH.md)**

- **`video`**: Input source (file, webcam, or URL) and output recording settings with codec selection
- **`display`**: Window settings, FPS limiting, and frame resizing options
- **`model`**: YOLO model path, device selection (CPU/GPU), confidence/IoU thresholds, and target classes
- **`tracking`**: BoT-SORT algorithm parameters including track thresholds, buffer size, matching thresholds, and ReID model path
- **`counter`**: Student counting confidence threshold and maximum confirmed students limit
- **`visualization`**: Overlay toggles (confidence, track ID, trajectory) and styling options (box thickness, font scale)
- **`performance`**: Frame skipping for faster processing, duration limits, and verbosity control
- **`statistics`**: Progress display and final statistics formatting

Every setting can be overridden via CLI flags such as `--input`, `--output`, `--model`, `--width`, `--height`, `--duration`, and `--no-display`.

## Project Layout

```
YOLO-Based-Student-Detection-and-Tracking/
├── main.py                      # Application entry point
├── config.yaml                  # Centralized configuration file
├── CONFIG_GUIDE.md              # Detailed configuration guide and parameter explanations
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── RESEARCH.md                  # Literature review and algorithm selection
│
├── models/                      # Pre-trained model weights
│   ├── yolo11s.pt              # YOLO11 small model for person detection
│   └── osnet_x0_25_msmt17.pt  # OSNet model for ReID appearance features
│
├── data/
│   ├── input/                  # Input video files
│   │   ├── classroom_1.mp4    # Sample classroom video
│   │   ├── classroom_2.mp4
│   │   ├── classroom_3.mp4
│   │   └── classroom_4.mp4
│   └── output/                 # Processed video output
│       └── tracked_output.mp4  # Annotated video with tracking results
│
└── src/
    ├── setup/                  # Setup and configuration
    │   ├── __init__.py        # Package initialization
    │   ├── cli.py             # Command-line argument parsing
    │   ├── config.py          # Configuration loading and CLI overrides
    │   └── components.py      # Component initialization
    │
    ├── app/                    # Application layer
    │   ├── pipeline.py        # Frame processing pipeline
    │   └── video_controller.py # Video loop, keyboard controls, progress reporting
    │
    ├── core/                   # Core algorithms
    │   ├── detector.py        # YOLO-based person detection wrapper
    │   ├── tracker.py         # BoT-SORT multi-object tracking implementation
    │   └── counter.py         # Student counting and statistics management
    │
    └── utils/                  # Utility functions
        ├── video.py           # Video I/O operations
        └── visualization.py   # Drawing functions for tracks, boxes, statistics
```

### Directory Descriptions

- **`setup/`** - CLI parsing, configuration loading, and component initialization
- **`app/`** - Frame processing pipeline and video playback controller
- **`core/`** - Core algorithms: detection (YOLO), tracking (BoT-SORT), and counting
- **`utils/`** - Video I/O and visualization utilities

---

## Demo

Here's a sample output showing the tracking system in action:

![Demo GIF](docs/demo.gif)

*Real-time student detection and tracking with bounding boxes, unique IDs, and trajectory visualization.*

**Download full video:** [tracked_output.mp4](data/output/tracked_output.mp4)
