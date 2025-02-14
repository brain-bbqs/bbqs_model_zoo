"""Get the prediction from the model."""
import click
from commands.ls import get_available
import dlclibrary as dlclib
from dlclibrary import download_huggingface_model, parse_available_supermodels
from bbqs_model_zoo.utils import (
    CACHE_PATH,
    MODELS_PATH,
    DATA_PATH,
)
from pathlib import Path
from dlclibrary.dlcmodelzoo import superanimal_configs
import os
import shutil

@click.command()
@click.option("--model", type=str, help="Name of model to evaluate on, in the form {model_type}/{model_name}")
@click.option("--dataset", type=str, help="Name of dataset to evaluate")
@click.option("--output_path", type=str, help="Path to output file")
def predict(model: str, dataset: str, output_path: str) -> None:
    """Get the prediction from the model."""

    model_args = model.split("/") # expect dlc, hf, or zenodo
    dataset_args = dataset.split("/")

    if check_exists(model, dataset):
        evaluate(model_args, dataset_args, output_path)
    
def check_exists(model, dataset):
    #TODO make error message clear
    #TODO clean this function

    model_args = model.split("/") # expect dlc, hf, or zenodo
    dataset_args = dataset.split("/")

    return len(model_args) == 3 and len(dataset_args) == 3 and model in get_available(model_args[0], "models") and dataset in get_available(dataset_args[0], "datasets")

#TODO def get_data(dataset):

def evaluate(model_args: list, dataset_args: list, output_path: str):
    model_cls, model_type, model_name = model_args
    data_cls, data_type, data_name = dataset_args

    #TODO data_path = get_data(dataset)

    if model_cls == "dlc":
        import deeplabcut

        # Ensure output directory exists
        target_dataset_path = os.path.join("/home/brukew/results", data_type)
        os.makedirs(target_dataset_path, exist_ok=True)

        # Define dataset path dynamically
        dataset_src = f"/home/brukew/bbqs_model_zoo/src/custom_data/{data_type}/{data_name}"
        dataset_dst = os.path.join(target_dataset_path, data_name)

        # Copy dataset to output path
        shutil.copy(dataset_src, dataset_dst)

        deeplabcut.video_inference_superanimal(
            [dataset_dst],
            model_type,
            videotype=".mov",
            scale_list=[200, 300, 400],
        )


        # available_models = dlclib.get_available_models(model_type)
        # if model_name not in available_models:
        #     raise ValueError(f"Model {model_name} not found in DLC.")

        # superanimal_configs = dlclib.parse_available_supermodels()
        # if model_type not in superanimal_configs:
        #     raise ValueError(f"Config {dataset} not found in superanimal_configs.")
            
        # config = superanimal_configs[model_type]

        # with config_path.open() as f:
        #     config = yaml.safe_load(f)

        # model_dir = Path(str(MODELS_PATH) + f"/{model_name}")
        # model_dir.mkdir()
        # model_path = model_dir / f"{model_type}_{model_name}.pt"

        # if not model_path.exists():
        #     print(f"Downloading model weights for {model_name}...")
        #     download_huggingface_model(model_type + "_" + model_name, model_dir)



    # can be any model on ls        