"""This module provides a commandline interface for listing available models."""

import click
import dlclibrary as dlclib
from huggingface_hub import HfApi
import requests

from bbqs_model_zoo.helper import OptionEatAll as OptionEatAll

_option_kwds = {"show_default": True}

def get_zenodo_datasets(q: str) -> list:
    """Collect Zenodo datasets"""
    url = f"https://zenodo.org/api/records/?q={q}"
    response = requests.get(url)
    data = response.json()
    zenodo_datasets = [f"zenodo/{dataset['metadata']['title'].replace(' ', '_')}" for dataset in data["hits"]["hits"]]

    return zenodo_datasets

def get_dlc_datasets() -> list:
    """Collect Hugging Face datasets."""
    api = HfApi()
    hf_datasets = api.list_datasets(author="mwmathis")
    dlc_datasets = [f"hf/{dataset.id}" for dataset in hf_datasets]
    dlc_datasets.extend(get_zenodo_datasets("maDLC%20AND%20Test%20AND%20creators.affiliation:EPFL"))

    return dlc_datasets

def get_hf_datasets() -> list:
    """Collect Hugging Face models."""
    api = HfApi()
    hf_datasets = api.list_datasets(search="pose-estimation")
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

def get_custom_datasets() -> list:
    """Collect custom datasets."""
    custom_datasets = []
    print("Custom datasets are not available yet.")
    return custom_datasets


@click.command()
@click.argument("category")
@click.option("--all", is_flag=True, default=False, help="List all models.")
@click.option(
    "--tool",
    type=str,
    multiple=True,
    help="tool name. {hf/dlc/custom}",
    **_option_kwds,
)
def ls(category: str, all: bool, tool: tuple, **kwrg: dict) -> None:
    """List available models.

    Examples:
        bbqs-zoo-cli ls datasets --all
        bbqs-zoo-cli ls models --all
        bbqs-zoo-cli ls models --tool dlc
        bbqs-zoo-cli ls models --tool dlc --tool hf ...
    """
    func_dict = {
            **dict.fromkeys(
                [
                    "dlc",
                    "deeplabcut",
                ],
                {"models": get_dlc_models, "datasets": get_dlc_datasets},
            ),
            **dict.fromkeys(
                ["hf", "huggingface", "hugging_face", "hugging-face"], {"models": get_hf_models, "datasets": get_hf_datasets}
            ),
            **dict.fromkeys(
                [
                    "custom",
                ],
                {"models": get_custom_models, "datasets": get_custom_datasets}
            ),
        }

    if category == "datasets":
        if all:
            tool = ["dlc", "hf", "custom"]

    if category == "models":
        if all:
            tool = ["dlc", "hf", "custom"] 

    for item in tool:
        for model in func_dict[item][category]():
            click.echo(model)
