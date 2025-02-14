import os
from pathlib import Path

if "BBQSZOO_CACHE" in os.environ:
    CACHE_PATH = Path(os.environ["BBQSZOO_CACHE"]).resolve() / ".bbqszoo"
else:
    CACHE_PATH = Path.home() / ".bbqszoo"

MODELS_PATH = CACHE_PATH / "models"
DATA_PATH = CACHE_PATH / "data"