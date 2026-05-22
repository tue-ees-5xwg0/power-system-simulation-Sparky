import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from power_grid_model import PowerGridModel
from power_grid_model import ComponentType
from power_grid_model import PowerGridModel, power_grid_meta_data
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
        self._model = self._initialize_model()
        self._pgm_batch_dataset = self._create_pgm_batch_dataset()


    def AggregateResults(self, *args, **kwargs) -> tuple[Dataset, Dataset]:
        preParseDataSet = self._RunModel(*args, **kwargs)
        node_results = self._output_table_row_per_timestamp(preParseDataSet)
        line_results = self._output_table_row_per_line(preParseDataSet)
        return node_results, line_results

    def _RunModel(self, *args, **kwargs) -> dict:
        # Create batch update dataset

        try:
            # Run time-series (batch) power flow calculation
            results = self._model.calculate_power_flow(
                *args, update_data=self._pgm_batch_dataset,
                symmetric=True  # standard for sym_load grids
                , **kwargs
            )
            return results

        except PowerGridError as e:
        # Pass through as required by assignment
            raise ValidationException("Batch dataset is invalid or power flow failed.") from e

    def _initialize_model(self) -> PowerGridModel:
        return PowerGridModel(self._power_grid_model_dataset)

    def _output_table_row_per_timestamp(self, preParseDataSet: dict) -> Dataset:
        timestamps = self._active_load_profiles.index
        node_results = []

        for i, ts in enumerate(timestamps):
            node_data = preParseDataSet["node"][i]

            node_voltages = node_data["voltages"]
            node_ids = node_data["id"]

            max_voltage = np.max(node_voltages)
            min_voltage = np.min(node_voltages)

            node_results.append({
                "Timestamp": ts,
                "Max_voltage": float(node_voltages[max_voltage]),
                "Max_voltage_node_id": int(node_ids[max_voltage]),
                "Min_voltage": float(node_voltages[min_voltage]),
                "Min_voltage_node_id": int(node_ids[min_voltage])
            })
        df_node_results = pd.DataFrame(node_results)
        df_node_results.set_index("Timestamp", inplace=True)

        return df_node_results

    def _output_table_row_per_line(self, preParseDataSet: dict) -> Dataset:
        line_data = preParseDataSet.get(ComponentType.line, preParseDataSet.get("line"))
        if line_data is None:
            raise ValueError("Line results not found in power flow output.")

        timestamps = pd.Index(self._active_load_profiles.index)
        if line_data.ndim == 1:
            line_data = line_data[np.newaxis, :]
        if line_data.shape[0] != len(timestamps):
            raise ValueError("Timestamp count does not match number of batch results.")

        p_loss = line_data["p_from"] + line_data["p_to"]
        loading = line_data["loading"]

        if len(timestamps) > 1:
            dt_hours = ((timestamps[1:] - timestamps[:-1]) / pd.Timedelta(hours=1)).to_numpy()
            total_loss_kwh = (0.5 * (p_loss[:-1] + p_loss[1:]) * dt_hours[:, None]).sum(axis=0) / 1000.0
        else:
            total_loss_kwh = np.zeros(line_data.shape[1], dtype=float)

        max_loading = np.full(line_data.shape[1], np.nan, dtype=float)
        min_loading = np.full(line_data.shape[1], np.nan, dtype=float)
        max_loading_ts = [pd.NaT] * line_data.shape[1]
        min_loading_ts = [pd.NaT] * line_data.shape[1]

        for idx in range(line_data.shape[1]):
            series = loading[:, idx]
            if np.all(np.isnan(series)):
                continue
            max_idx = int(np.nanargmax(series))
            min_idx = int(np.nanargmin(series))
            max_loading[idx] = series[max_idx]
            min_loading[idx] = series[min_idx]
            max_loading_ts[idx] = timestamps[max_idx]
            min_loading_ts[idx] = timestamps[min_idx]

        line_ids = line_data[0]["id"]
        result = pd.DataFrame(
            {
                "Line_ID": line_ids,
                "Total_loss": total_loss_kwh,
                "Max_loading": max_loading,
                "Max_loading_Timestamp": max_loading_ts,
                "Min_loading": min_loading,
                "Min_loading_Timestamp": min_loading_ts,
            }
        ).set_index("Line_ID")

        return result

    def _create_pgm_batch_dataset(self) -> dict:
        timestamps = self._active_load_profiles.index
        load_ids = [col for col in self._active_load_profiles.columns if col != 'Timestamp']

        update_meta = power_grid_meta_data["update"]["sym_load"]
        sym_load_dtype = update_meta.dtype
        status_nan = update_meta.nan_scalar["status"][0]

        sym_load_updates = []
        for ts in timestamps:
            ts_updates = []
            for load_id in load_ids:
                ts_updates.append(
                    (
                        int(load_id),
                        status_nan,
                        float(self._active_load_profiles.loc[ts, load_id]),
                        float(self._reactive_load_profiles.loc[ts, load_id]),
                    )
                )
            sym_load_updates.append(np.array(ts_updates, dtype=sym_load_dtype))

        return {"sym_load": np.stack(sym_load_updates, axis=0)}

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
