"""GestureSlide configuration: paths, gesture classes and key mappings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "gestures.csv"
MODEL_PATH = ROOT / "model" / "gestures.keras"
LABELS_PATH = ROOT / "model" / "labels.json"
METRICS_PATH = ROOT / "model" / "metrics.json"
LOG_DIR = ROOT / "logs"

# name -> (record key, display name, keyboard key sent, description)
# The first three are the core prototype gestures from the project guide.
# "peace" is the optional extra gesture: record it only if you want it
# (training adapts automatically to however many classes are in the CSV).
GESTURES = {
    "open_palm": ("1", "Open palm", "right", "Next slide"),
    "fist": ("2", "Closed fist", "left", "Previous slide"),
    "thumbs_up": ("3", "Thumbs up", "f5", "Start slideshow"),
    "peace": ("4", "Peace sign (extra)", "b", "Blank / unblank screen"),
}

# Safety: with --control, keys are only sent if the active window title
# contains one of these words (disable with --no-guard).
GUARD_KEYWORDS = ["powerpoint", "slide show", "slideshow", "impress", "presentation"]

# Real-time defaults (all overridable from the command line)
DEFAULT_THRESHOLD = 0.85   # minimum softmax confidence
DEFAULT_STABLE = 6         # consecutive frames that must agree
DEFAULT_COOLDOWN = 1.2     # seconds between commands
SAMPLES_TARGET = 150       # suggested samples per class
