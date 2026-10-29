"""Small helpers shared by every script: paths, logging, seeds, tag loading.

Keeping these in one place means every script logs the same way and agrees on
where files live.
"""

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Repo root = the folder containing src/. Using __file__ makes paths work no
# matter which directory you run the command from.
ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
DATA = ROOT / "data"
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

SEED = 1234


def get_logger(script_name: str) -> logging.Logger:
    """Log to the console AND to logs/<script>_<timestamp>.log.

    Every run gets its own timestamped file, so you can always go back and see
    exactly what a past run printed.
    """
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS / f"{script_name}_{stamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate lines if called twice

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)  # console: the essentials
    console.setFormatter(fmt)

    file = logging.FileHandler(log_path, encoding="utf-8")
    file.setLevel(logging.DEBUG)  # file: everything, including DEBUG details
    file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file)
    logger.info("logging to %s", log_path.relative_to(ROOT))
    logger.info("command: %s", " ".join(sys.argv))
    return logger


def set_seeds(seed: int = SEED) -> None:
    """Fix every random number generator we use, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_tags() -> dict:
    """Read config/tags.yaml -> {"themes": [...], "images": [...]}."""
    with open(CONFIG / "tags.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict]:
    """JSONL = one JSON object per line. Easy to append to and to grep."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
