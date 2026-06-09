import numpy as np
import pandas as pd
from pandas import DataFrame
from power_grid_model import ComponentType, PowerGridModel, initialize_array
from power_grid_model.errors import PowerGridError

from power_system_simulation.validate import (
    ValidationException,
    _validate_active_reactive_profiles,
    _validate_power_grid_model,
)


class GridModel:
    def __init__(self, power_grid_model_path: str, active_load_profiles_path: str, reactive_load_profiles_path: str):
        self._power_grid_model_dataset = _validate_power_grid_model(power_grid_model_path)
        self._active_load_profiles, self._reactive_load_profiles = _validate_active_reactive_profiles(
            active_load_profiles_path, reactive_load_profiles_path
        )
        self._model = self._initialize_model()
        self._pgm_batch_dataset = self._create_pgm_batch_dataset()

    def AggregateResults(self, *args, **kwargs) -> tuple[DataFrame, DataFrame]:
        preParseDataSet = self._RunModel(*args, **kwargs)
        node_results = self._output_table_row_per_timestamp(preParseDataSet)
        line_results = self._output_table_row_per_line(preParseDataSet)
        return node_results, line_results

    def _RunModel(self, *args, **kwargs) -> dict:
        # Create batch update dataset

        try:
            # Run time-series (batch) power flow calculation
            results = self._model.calculate_power_flow(
                *args,
                update_data=self._pgm_batch_dataset,
                symmetric=True,  # standard for sym_load grids
                **kwargs,
            )
            return results

        except PowerGridError as e:
            # Pass through as required by assignment
            raise ValidationException("Batch dataset is invalid or power flow failed.") from e

    def _initialize_model(self) -> PowerGridModel:
        return PowerGridModel(self._power_grid_model_dataset)

    def _output_table_row_per_timestamp(self, preParseDataSet: dict) -> DataFrame:
        node_data = preParseDataSet.get(ComponentType.node, preParseDataSet.get("node"))
        if node_data is None:
            raise ValueError("Node results not found in power flow output.")

        timestamps = pd.to_datetime(self._active_load_profiles.index)
        if node_data.ndim == 1:
            node_data = node_data[np.newaxis, :]
        if node_data.shape[0] != len(timestamps):
            raise ValueError("Timestamp count does not match number of batch results.")

        voltages = node_data["u_pu"]
        node_ids = node_data[0]["id"]

        max_voltage = np.full(len(timestamps), np.nan, dtype=float)
        min_voltage = np.full(len(timestamps), np.nan, dtype=float)
        max_voltage_node = np.full(len(timestamps), np.nan, dtype=float)
        min_voltage_node = np.full(len(timestamps), np.nan, dtype=float)

        for idx in range(len(timestamps)):
            series = voltages[idx]
            if np.all(np.isnan(series)):
                continue
            max_idx = int(np.nanargmax(series))
            min_idx = int(np.nanargmin(series))
            max_voltage[idx] = series[max_idx]
            min_voltage[idx] = series[min_idx]
            max_voltage_node[idx] = node_ids[max_idx]
            min_voltage_node[idx] = node_ids[min_idx]

        result = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Max_Voltage": max_voltage,
                "Max_Voltage_Node": max_voltage_node,
                "Min_Voltage": min_voltage,
                "Min_Voltage_Node": min_voltage_node,
            }
        ).set_index("Timestamp")

        return result

    def _output_table_row_per_line(self, preParseDataSet: dict) -> DataFrame:
        line_data = preParseDataSet.get(ComponentType.line, preParseDataSet.get("line"))
        if line_data is None:
            raise ValueError("Line results not found in power flow output.")

        timestamps = pd.to_datetime(self._active_load_profiles.index)
        if line_data.ndim == 1:
            line_data = line_data[np.newaxis, :]
        if line_data.shape[0] != len(timestamps):
            raise ValueError("Timestamp count does not match number of batch results.")

        p_loss = line_data["p_from"] + line_data["p_to"]
        loading = line_data["loading"]

        if len(timestamps) > 1:
            dt_hours = (timestamps[1:] - timestamps[:-1]).total_seconds().to_numpy() / 3600.0
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
                "Total_Loss": total_loss_kwh,
                "Max_Loading": max_loading,
                "Max_Loading_Timestamp": max_loading_ts,
                "Min_Loading": min_loading,
                "Min_Loading_Timestamp": min_loading_ts,
            }
        ).set_index("Line_ID")

        return result

    def _create_pgm_batch_dataset(self) -> dict:
        timestamps = self._active_load_profiles.index
        load_ids = [col for col in self._active_load_profiles.columns if col != "Timestamp"]

        num_timestamps = len(timestamps)
        num_loads = len(load_ids)

        # 1. Automatically create a 2D array pre-filled with the correct NaN values
        sym_load_updates = initialize_array("update", "sym_load", (num_timestamps, num_loads))

        # 2. Fill in the specific data using vectorized assignment
        sym_load_updates["id"] = [int(load_id) for load_id in load_ids]
        sym_load_updates["p_specified"] = self._active_load_profiles[load_ids].to_numpy()
        sym_load_updates["q_specified"] = self._reactive_load_profiles[load_ids].to_numpy()

        return {"sym_load": sym_load_updates}
