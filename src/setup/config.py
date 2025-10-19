"""
Configuration Management Module

This module handles loading and managing configuration from YAML files and CLI arguments.
It provides functions to load configuration files, apply CLI overrides, and define
minimal default settings when no configuration file is provided.

The configuration system allows users to define settings in a YAML file and override
them via command-line arguments for quick testing and experimentation.

Example:
    >>> config = load_config("config.yaml")
    >>> updated_config = apply_cli_overrides(config, {"input_source": 0})
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml


# Minimal default configuration used when no config file is provided
# These defaults ensure the application can run with basic settings
MINIMAL_DEFAULTS: dict[str, Any] = {
    "video": {
        "input_source": 0,  # Default to webcam (index 0)
    },
    "model": {
        "path": "models/yolo11s.pt",  # Default YOLO model
        "device": "cpu",  # Default to CPU processing
    },
}


def load_config(config_path: str | None) -> dict[str, Any]:
    """Load configuration from a YAML file.
    
    This function reads a YAML configuration file and returns its contents as a
    dictionary. If no config path is provided or the file doesn't exist, it returns
    minimal default settings to ensure the application can still run.
    
    Args:
        config_path (str | None): Path to the YAML configuration file.
                                  If None, returns minimal defaults.
    
    Returns:
        dict[str, Any]: Configuration dictionary with all settings.
                       Returns MINIMAL_DEFAULTS if no config file is provided.
    
    Raises:
        RuntimeError: If the config file exists but cannot be loaded due to
                     parsing errors or other issues.
    
    Example:
        >>> config = load_config("config.yaml")
        >>> print(config["model"]["path"])
        'models/yolo11s.pt'
        
        >>> # No config file - uses defaults
        >>> config = load_config(None)
        >>> print(config["video"]["input_source"])
        0
    """
    # If no config path provided, return minimal defaults
    if not config_path:
        return deepcopy(MINIMAL_DEFAULTS)

    # Try to load and parse the YAML file
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        # File doesn't exist - use defaults
        return deepcopy(MINIMAL_DEFAULTS)
    except Exception as exc:
        # File exists but has errors - raise exception
        raise RuntimeError(f"Failed to load configuration '{config_path}': {exc}") from exc

    return loaded


def apply_cli_overrides(config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Apply command-line argument overrides to configuration.
    
    This function takes the loaded configuration and applies any CLI argument
    overrides. This allows users to test different settings without modifying
    the configuration file.
    
    The function creates a deep copy of the config to avoid modifying the original,
    then applies each CLI option that was provided.
    
    Args:
        config (dict[str, Any]): Original configuration dictionary loaded from YAML.
        options (dict[str, Any]): CLI options dictionary from parse_cli_args().
    
    Returns:
        dict[str, Any]: Updated configuration with CLI overrides applied.
    
    Example:
        >>> config = load_config("config.yaml")
        >>> cli_options = {"input_source": 0, "verbose": True}
        >>> updated = apply_cli_overrides(config, cli_options)
        >>> print(updated["video"]["input_source"])
        0
        >>> print(updated["performance"]["verbose"])
        True
    """
    # Create a deep copy to avoid modifying the original config
    updated = deepcopy(config)

    # Override video input source (webcam index, file path, or URL)
    if options.get("input_source") is not None:
        updated.setdefault("video", {})["input_source"] = options["input_source"]

    # Override output video path and enable output
    if options.get("output_path"):
        video_output = updated.setdefault("video", {}).setdefault("output", {})
        video_output["enabled"] = True
        video_output["path"] = options["output_path"]

    # Override YOLO model path
    if options.get("model_path"):
        updated.setdefault("model", {})["path"] = options["model_path"]

    # Disable display window
    if options.get("disable_display"):
        updated.setdefault("display", {})["enabled"] = False

    # Override display width
    if options.get("width"):
        resize = updated.setdefault("display", {}).setdefault("resize", {})
        resize["enabled"] = True
        resize["width"] = options["width"]

    # Override display height
    if options.get("height"):
        resize = updated.setdefault("display", {}).setdefault("resize", {})
        resize["enabled"] = True
        resize["height"] = options["height"]

    # Apply performance settings (duration limit and verbosity)
    performance = updated.setdefault("performance", {})
    if options.get("duration") is not None:
        performance["max_duration"] = options["duration"]
    if options.get("verbose"):
        performance["verbose"] = True

    return updated

