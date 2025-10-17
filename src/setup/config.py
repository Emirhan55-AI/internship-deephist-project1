from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml


MINIMAL_DEFAULTS: dict[str, Any] = {
    "video": {
        "input_source": 0,
    },
    "model": {
        "path": "models/yolo11s.pt",
        "device": "cpu",
    },
}


def load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return deepcopy(MINIMAL_DEFAULTS)

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return deepcopy(MINIMAL_DEFAULTS)
    except Exception as exc:
        raise RuntimeError(f"Failed to load configuration '{config_path}': {exc}") from exc

    return loaded


def apply_cli_overrides(config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)

    if options.get("input_source") is not None:
        updated.setdefault("video", {})["input_source"] = options["input_source"]

    if options.get("output_path"):
        video_output = updated.setdefault("video", {}).setdefault("output", {})
        video_output["enabled"] = True
        video_output["path"] = options["output_path"]

    if options.get("model_path"):
        updated.setdefault("model", {})["path"] = options["model_path"]

    if options.get("disable_display"):
        updated.setdefault("display", {})["enabled"] = False

    if options.get("width"):
        resize = updated.setdefault("display", {}).setdefault("resize", {})
        resize["enabled"] = True
        resize["width"] = options["width"]

    if options.get("height"):
        resize = updated.setdefault("display", {}).setdefault("resize", {})
        resize["enabled"] = True
        resize["height"] = options["height"]

    performance = updated.setdefault("performance", {})
    if options.get("duration") is not None:
        performance["max_duration"] = options["duration"]
    if options.get("verbose"):
        performance["verbose"] = True

    return updated

