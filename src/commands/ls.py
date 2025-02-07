"""This module provides a commandline interface for listing available models."""

import click
import dlclibrary as dlclib
from huggingface_hub import HfApi

from bbqs_model_zoo.helper import OptionEatAll as OptionEatAll

_option_kwds = {"show_default": True}


def get_hf_datasets() -> list:
    """Collect Hugging Face datasets."""
    api = HfApi()
    hf_datasets = api.list_datasets(author="mwmathis")
    hf_datasets = [f"hf/{dataset.id}" for dataset in hf_datasets]

    return hf_datasets


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


def get_custom_models() -> list:
    """Collect custom models."""
    custom_models = []
    print("Custom models are not available yet.")
    return custom_models


@click.command()
@click.argument("type")
@click.option("--all", is_flag=True, help="List all models.")
@click.option(
    "--tool",
    type=str,
    multiple=True,
    help="tool name. {hf/dlc/custom}",
    **_option_kwds,
)
def ls(type: str, all: bool, tool: str, **kwrg: dict) -> None:
    """List available models.

    Examples:
        bbqs-zoo-cli ls datasets --all
        bbqs-zoo-cli ls models --all
        bbqs-zoo-cli ls models --tool dlc
        bbqs-zoo-cli ls models --tool dlc --tool hf ...
    """
    if type == "datasets":
        for dataset in get_hf_datasets():
            click.echo(dataset)
        return

    if type == "models":
        if all:
            tool = ["dlc", "hf", "custom"]

        func_dict = {
            **dict.fromkeys(
                [
                    "dlc",
                    "deeplabcut",
                ],
                get_dlc_models,
            ),
            **dict.fromkeys(
                ["hf", "huggingface", "hugging_face", "hugging-face"], get_hf_models
            ),
            **dict.fromkeys(
                [
                    "custom",
                ],
                get_custom_models,
            ),
        }

        for item in tool:
            for model in func_dict[item]():
                click.echo(model)
