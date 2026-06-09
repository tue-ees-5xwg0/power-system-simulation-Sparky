import os

import pandas as pd
from pandas import DataFrame
from power_grid_model._core.data_types import Dataset
from power_grid_model.errors import PowerGridError
from power_grid_model.utils import json_deserialize


class ValidationException(Exception):
    pass


class ProfilesNotMatchingError(Exception):
    pass

def _validate_power_grid_model(power_grid_model_path: str) -> Dataset:
    # check string is not empty
    if not power_grid_model_path:
        raise ValidationException("Power grid model path is required.")
    # check file exists
    if os.path.isfile(power_grid_model_path) is False:
        raise ValidationException(f"Power grid model file not found: {power_grid_model_path}")
    # check file extension is json
    if not power_grid_model_path.endswith(".json"):
        raise ValidationException(f"Power grid model file must be a JSON file: {power_grid_model_path}")
    # try to deserialize file
    try:
        with open(power_grid_model_path) as f:
            power_grid_model_data = json_deserialize(f.read())
        return power_grid_model_data
    except ValueError as e:
        raise ValidationException("Power grid model data is inconsistent or a component is unknown.") from e
    except PowerGridError as e:
        raise ValidationException("There was a internal error in the power grid model.") from e


def _validate_active_reactive_profiles(
    active_load_profiles_path: str, reactive_load_profiles_path: str
) -> tuple[DataFrame, DataFrame]:
    active_load_profiles = _validate_load_profile("Active", active_load_profiles_path)
    reactive_load_profiles = _validate_load_profile("Reactive", reactive_load_profiles_path)
    _validate_profiles_match(active_load_profiles, reactive_load_profiles)
    return active_load_profiles, reactive_load_profiles


def _validate_load_profile(Type: str, load_profiles_path: str) -> DataFrame:
    # check string is not empty
    if not load_profiles_path:
        raise ValidationException(f"{Type} load profiles path is required.")
    # check files exist
    if os.path.isfile(load_profiles_path) is False:
        raise ValidationException(f"{Type} load profiles file not found: {load_profiles_path}")
    # check file extensions are parquet
    if not load_profiles_path.endswith(".parquet"):
        raise ValidationException(f"{Type} load profiles file must be a Parquet: {load_profiles_path}")
    # try to read files
    try:
        load_profiles = pd.read_parquet(load_profiles_path)
    except Exception as e:
        raise ValidationException(f"Error occurred while reading {Type} load profiles Parquet.") from e

    return load_profiles


def _validate_profiles_match(active_load_profiles: DataFrame, reactive_load_profiles: DataFrame) -> None:
    if active_load_profiles.shape != reactive_load_profiles.shape:
        raise ProfilesNotMatchingError("Active and reactive load profiles do not match.")

    # Check if ID's match
    if not active_load_profiles.columns.equals(reactive_load_profiles.columns):
        raise ProfilesNotMatchingError("Active and reactive load profiles have mismatched Load IDs.")

    # Check if timestamps match
    if not active_load_profiles.index.equals(reactive_load_profiles.index):
        raise ProfilesNotMatchingError("Active and reactive load profiles have mismatched timestamps.")

def _validate_ev_profile(active_load_profiles: DataFrame, ev_pool: DataFrame) -> None:
    #Check if the ev profile match the loads
    if not active_load_profiles.index.equals(ev_pool.index):
        raise ProfilesNotMatchingError("Active load profile and EV profile have different time indices.")

    #Check if there are enough columns in the ev profile
    if len(ev_pool.columns) < len(active_load_profiles.columns):
        raise ProfilesNotMatchingError("EV profile has fewer columns than the active load profile.")

def _validate_transformer(dataset: Dataset, transformer_id: int) -> None:
    # Check if the transformer id is valid
    if transformer_id not in dataset["transformer"]["id"]:
        raise ValidationException(f"Transformer ID {transformer_id} is not valid.")
    # check if there is only 1 transformer in the system
    if len(dataset["transformer"]) != 1:
        raise ValidationException(
            "There should only be one transformer in the system. Please specify the transformer ID.")

# There should be exactly one source in the system
def _validate_source(dataset: Dataset) -> None:
    if len(dataset["source"]) != 1:
        raise ValidationException("There should be exactly one source in the system.")

# check if Every LV feeder ID is a valid line ID.
def _validate_feeder_line_ids(dataset: Dataset, feeder_line_ids: list[int]) -> None:
    for line_id in feeder_line_ids:
        if line_id not in dataset["line"]["id"]:
            raise ValidationException(f"Feeder line ID {line_id} is not valid.")

# check if Every feeder line has from_node == transformer.to_node
def _validate_feeder_connections(dataset: Dataset, feeder_line_ids: list[int]) -> None:
    transformer_to_node = dataset["transformer"]["to_node"][0]
    line_from_node_dict = dict(zip(dataset["line"]["id"], dataset["line"]["from_node"], strict=True))
    for line_id in feeder_line_ids:
        if line_from_node_dict.get(line_id) != transformer_to_node:
            raise ValidationException(f"Feeder line ID {line_id} is not connected to the transformer.")
