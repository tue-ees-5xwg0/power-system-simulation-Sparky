import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from power_grid_model._core.data_types import Dataset
from power_grid_model.errors import PowerGridError
from power_grid_model.utils import json_deserialize


class ValidationException(Exception):
    pass


class ProfilesNotMatchingError(Exception):
    pass


class GridModel:
    def __init__(self, power_grid_model_path: str, active_load_profiles_path: str, reactive_load_profiles_path: str):
        self._power_grid_model_dataset = _validate_power_grid_model(power_grid_model_path)
        self._active_load_profiles, self._reactive_load_profiles = _validate_active_reactive_profiles(
            active_load_profiles_path, reactive_load_profiles_path
        )

    def _create_pgm_batch_dataset(self) -> dict:
        timestamps = self._active_load_profiles.index
        load_ids = [col for col in self._active_load_profiles.columns if col != 'Timestamp']

        sym_load_updates = []
        for ts in timestamps:
            ts_updates = []
            for load_id in load_ids:
                ts_updates.append({
                    "id": int(load_id),
                    "p_specified": float(self._active_load_profiles.loc[ts, load_id]),
                    "q_specified": float(self._reactive_load_profiles.loc[ts, load_id])
                })
            sym_load_updates.append(np.array(ts_updates, dtype=self.model.get_component_type("sym_load")))

        return {"sym_load": sym_load_updates}

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
