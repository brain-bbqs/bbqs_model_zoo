"""This module provides a commandline interface for listing available models."""

import click
import dlclibrary as dlclib
from huggingface_hub import HfApi

from bbqs_model_zoo.helper import OptionEatAll as OptionEatAll

_option_kwds = {"show_default": True}


def get_hf_models() -> list:
    """Collect Hugging Face models."""
    api = HfApi()
    hf_models = api.list_models(filter="pose-estimation")
    hf_models = [f"hf/{model.modelId}" for model in hf_models]

    return hf_models


def get_dlc_models() -> list:
    """Collect DeepLabCut models."""
    dlc_models = []
    dlc_datasets = dlclib.get_available_datasets()
    for dlc_dataset in dlc_datasets:
        models = [
            f"dlc/{dlc_dataset}/{item}"
            for item in dlclib.get_available_models(dlc_dataset)
        ]
        dlc_models.extend(models)

    return dlc_models


@click.command()
@click.option("--all", is_flag=True, help="List all models.")
@click.option(
    "--tool",
    type=str,
    multiple=True,
    help="tool name. should be in the form of <org>/<model>/<version>",
    **_option_kwds,
)
def ls(all: bool, tool: str) -> None:
    """List available models.

    Examples:
        bbqs-zoo-cli ls --all
        bbqs-zoo-cli ls --tool dlc
        bbqs-zoo-cli ls --tool dlc --tool hf ...
    """
    if all:
        hf_models = get_hf_models()
        dlc_models = get_dlc_models()
        models = hf_models + dlc_models

        for item in models:
            click.echo(item)

    for item in tool:
        if item in ["dlc", "deeplabcut"]:
            dlc_models = get_dlc_models()
            for dlc_model in dlc_models:
                click.echo(dlc_model)

        if item in ["hf", "huggingface", "hugging_face", "hugging-face"]:
            hf_models = get_hf_models()
            for hf_model in hf_models:
                click.echo(hf_model)
