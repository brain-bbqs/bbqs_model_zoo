"""Initialize."""
import click
from bbqs_model_zoo.utils import (
    CACHE_PATH,
    MODELS_PATH,
    DATA_PATH,
)
import os
from pathlib import Path

@click.command()
@click.option("-c", "--cache", default=None)
def init(cache) -> None:
    """Initialize."""
    global CACHE_PATH, MODELS_PATH, DATA_PATH
    
    # env variable takes precedence, then looks to passed in cache path
    if "BBQSZOO_CACHE" in os.environ:
        env_cache = Path(os.environ["BBQSZOO_CACHE"]).expanduser().resolve()
        CACHE_PATH = env_cache / ".bbqszoo"
    elif cache is not None:
        CACHE_PATH = Path(cache).expanduser().resolve() / ".bbqszoo"
    else:
        CACHE_PATH = CACHE_PATH

    os.makedirs(CACHE_PATH, exist_ok=True)

    MODELS_PATH = CACHE_PATH / "models"
    DATA_PATH = CACHE_PATH / "data"
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

    click.echo(
        f"Using cache directory: {CACHE_PATH}\n\n"
        "You can change the cache location by:\n"
        "  1) Setting the environment variable: export BBQSZOO_CACHE=/path/to/cache\n"
        "  2) Or by running: nobrainer-zoo init --cache /path/to/cache\n"
        "Note that the 'BBQSZOO_CACHE' environment variable overrides the --cache option."
    )
