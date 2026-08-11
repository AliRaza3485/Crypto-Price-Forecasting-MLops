"""
Central configuration loader.

Reads config/config.yaml once and exposes it as CONFIG. The project root is
discovered from THIS file's own location, so every path in the config can be
turned into an absolute path via get_path() — meaning the code works no matter
which directory you run it from, and nothing is hard-coded.
"""

from pathlib import Path

import yaml

# This file lives at <project_root>/src/config.py, so the root is two levels up:
#   parents[0] = .../src   ,   parents[1] = .../project_root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def get_path(relative_path: str) -> Path:
    """Turn a project-relative path (from the config) into an absolute path."""
    return PROJECT_ROOT / relative_path
