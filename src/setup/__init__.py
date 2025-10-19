"""
Setup Package

This package provides initialization and configuration utilities for the Eyes on You
application. It handles command-line argument parsing, configuration loading, and
component initialization.

The package exports the following functions:
- parse_cli_args: Parse command-line arguments
- load_config: Load configuration from YAML file
- apply_cli_overrides: Apply CLI overrides to configuration
- setup_components: Initialize all application components

Example:
    >>> from src.setup import parse_cli_args, load_config, setup_components
    >>> options = parse_cli_args()
    >>> config = load_config(options["config_path"])
    >>> pipeline, video, counter, tracker = setup_components(config, logger)
"""

from src.setup.cli import parse_cli_args
from src.setup.config import apply_cli_overrides, load_config
from src.setup.components import setup_components

__all__ = ["parse_cli_args", "load_config", "apply_cli_overrides", "setup_components"]

