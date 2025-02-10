"""Get the prediction from the model."""
import click
from commands.ls import get_dlc_models, get_hf_models, get_custom_models
import dlclibrary as dlclib
from dlclibrary import download_huggingface_model


@click.command()
@click.option("--model", type=str, help="Name of model to evaluate on, in the form {model_type}/{model_name}")
@click.option("--dataset", type=str, help="Name of dataset to evaluate")
@click.option("--output_path", type=str, help="Path to output file")
def predict(model: str, dataset: str, output_path: str) -> None:
    """Get the prediction from the model."""

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

    model_args = model.split("/") # expect dlc, hf, or zenodo

    if model_args[0] in func_dict:
        models = func_dict[model_args[0]]()
        if model in models:
            evaluate(model_args, dataset)
    

def evaluate(model_args: list, dataset: str):
    if model_args[0] == "dlc":
        available_models = dlclib.get_available_models(model_args[1])
        if model_args[2] not in available_models:
            raise ValueError(f"Model {model} not found in DLC.")

        # Run inference with DLC
        