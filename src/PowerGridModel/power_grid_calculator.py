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


class _InputValidator:
    @staticmethod
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

    @staticmethod
    def _validate_load_profiles(
        active_load_profiles_path: str, reactive_load_profiles_path: str
    ) -> tuple[DataFrame, DataFrame]:
        # check string is not empty
        if not active_load_profiles_path:
            raise ValidationException("Active load profiles path is required.")
        if not reactive_load_profiles_path:
            raise ValidationException("Reactive load profiles path is required.")
        # check files exist
        if os.path.isfile(active_load_profiles_path) is False:
            raise ValidationException(f"Active load profiles file not found: {active_load_profiles_path}")
        if os.path.isfile(reactive_load_profiles_path) is False:
            raise ValidationException(f"Reactive load profiles file not found: {reactive_load_profiles_path}")
        # check file extensions are parquet
        if not active_load_profiles_path.endswith(".parquet"):
            raise ValidationException(f"Active load profiles file must be a Parquet: {active_load_profiles_path}")
        if not reactive_load_profiles_path.endswith(".parquet"):
            raise ValidationException(f"Reactive load profiles file must be a Parquet: {reactive_load_profiles_path}")
        # try to read files
        try:
            active_load_profiles = pd.read_parquet(active_load_profiles_path)
        except Exception as e:
            raise ValidationException("Error occurred while reading active load profiles Parquet.") from e
        try:
            reactive_load_profiles = pd.read_parquet(reactive_load_profiles_path)
        except Exception as e:
            raise ValidationException("Error occurred while reading reactive load profiles Parquet.") from e
        # check profiles match
        if active_load_profiles.shape != reactive_load_profiles.shape:
            raise ProfilesNotMatchingError("Active and reactive load profiles do not match.")

        return active_load_profiles, reactive_load_profiles



class GridModel(_InputValidator):
    def __init__(
        self,
        power_grid_model_path: str,
        active_load_profiles_path: str = None,
        reactive_load_profiles_path: str = None,
    ):
        self.power_grid_model = self._validate_power_grid_model(power_grid_model_path)
        self.active_load_profiles, self.reactive_load_profiles = self._validate_load_profiles(
            active_load_profiles_path, reactive_load_profiles_path
        )
