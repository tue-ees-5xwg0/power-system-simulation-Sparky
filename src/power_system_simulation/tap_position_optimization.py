from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from os import cpu_count
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from power_grid_model import ComponentType, PowerGridModel, power_grid_meta_data
from power_grid_model._core.data_types import Dataset
from power_grid_model.errors import PowerGridError
from power_grid_model import initialize_array

TapOptimizationCriterion = Callable[[pd.DataFrame, pd.DataFrame], float]


class TapOptimizationError(Exception):
    pass


@dataclass(frozen=True)
class TapOptimizationResult:
    tap_position: int
    criterion: TapOptimizationCriterion
    criterion_value: float
    total_loss_kwh: float
    average_voltage_deviation_pu: float
    all_tap_results: pd.DataFrame


@dataclass(frozen=True)
class _TapPositionEvaluation:
    tap_position: int
    timestamp_table: pd.DataFrame
    line_table: pd.DataFrame
    total_loss_kwh: float
    average_voltage_deviation_pu: float


def minimize_total_loss(timestamp_table: pd.DataFrame, line_table: pd.DataFrame) -> float:
    return float(line_table["Total_Loss"].sum())


def minimize_average_voltage_deviation(timestamp_table: pd.DataFrame, line_table: pd.DataFrame) -> float:
    return float(
        ((timestamp_table["Max_Voltage"] - 1.0).abs() + (timestamp_table["Min_Voltage"] - 1.0).abs()).mean() / 2
    )


def _evaluate_tap_position_candidate(
    tap_position: int,
    dataset: Dataset,
    active_load_profiles: pd.DataFrame,
    reactive_load_profiles: pd.DataFrame,
    tap_transformer_id: int,
    power_flow_args: tuple[Any, ...],
    power_flow_kwargs: dict[str, Any],
) -> _TapPositionEvaluation:
    candidate_dataset = _copy_dataset_with_tap_position(dataset, tap_transformer_id, tap_position)
    results = _run_time_series_power_flow(
        candidate_dataset,
        active_load_profiles,
        reactive_load_profiles,
        *power_flow_args,
        **power_flow_kwargs,
    )
    timestamp_table = _output_table_row_per_timestamp(results, active_load_profiles)
    line_table = _output_table_row_per_line(results, active_load_profiles)

    return _TapPositionEvaluation(
        tap_position=tap_position,
        timestamp_table=timestamp_table,
        line_table=line_table,
        total_loss_kwh=minimize_total_loss(timestamp_table, line_table),
        average_voltage_deviation_pu=minimize_average_voltage_deviation(timestamp_table, line_table),
    )


def _copy_dataset_with_tap_position(dataset: Dataset, tap_transformer_id: int, tap_position: int) -> Dataset:
    copied_dataset = {component_type: component.copy() for component_type, component in dataset.items()}
    transformer = _get_component_from_dataset(copied_dataset, "transformer")
    matches = transformer["id"] == tap_transformer_id
    transformer["tap_pos"][matches] = tap_position
    return copied_dataset


def _run_time_series_power_flow(
    dataset: Dataset,
    active_load_profiles: pd.DataFrame,
    reactive_load_profiles: pd.DataFrame,
    *args,
    **kwargs,
) -> dict:
    try:
        model = PowerGridModel(dataset)
        power_flow_kwargs = {"threading": 1, **kwargs}
        return model.calculate_power_flow(
            *args,
            update_data=_create_pgm_batch_dataset(active_load_profiles, reactive_load_profiles),
            symmetric=True,
            **power_flow_kwargs,
        )
    except PowerGridError as error:
        raise TapOptimizationError("Power flow failed during tap position optimization.") from error


def _create_pgm_batch_dataset(
    active_load_profiles: pd.DataFrame,
    reactive_load_profiles: pd.DataFrame,
) -> dict:
    timestamps = active_load_profiles.index
    load_ids = list(active_load_profiles.columns)

    update_meta = power_grid_meta_data["update"]["sym_load"]
    sym_load_dtype = update_meta.dtype
    status_nan = update_meta.nan_scalar["status"][0]

    timestamps = active_load_profiles.index
    load_ids = list(active_load_profiles.columns)

    num_timestamps = len(timestamps)
    num_loads = len(load_ids)

    # Initialize the update array with correct shape and NaN values

    sym_load_updates = initialize_array("update", "sym_load", (num_timestamps, num_loads))

    # Fill in the load data
    sym_load_updates["id"] = [int(load_id) for load_id in load_ids]
    sym_load_updates["p_specified"] = active_load_profiles[load_ids].to_numpy()
    sym_load_updates["q_specified"] = reactive_load_profiles[load_ids].to_numpy()

    return {"sym_load": sym_load_updates}


def _output_table_row_per_timestamp(power_flow_results: dict, active_load_profiles: pd.DataFrame) -> DataFrame:
    node_data = power_flow_results.get(ComponentType.node, power_flow_results.get("node"))
    if node_data is None:
        raise TapOptimizationError("Node results not found in power flow output.")

    timestamps = pd.Index(active_load_profiles.index)
    if node_data.ndim == 1:
        node_data = node_data[np.newaxis, :]
    if node_data.shape[0] != len(timestamps):
        raise TapOptimizationError("Timestamp count does not match number of batch results.")

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

    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Max_Voltage": max_voltage,
            "Max_Voltage_Node": max_voltage_node,
            "Min_Voltage": min_voltage,
            "Min_Voltage_Node": min_voltage_node,
        }
    ).set_index("Timestamp")


def _output_table_row_per_line(power_flow_results: dict, active_load_profiles: pd.DataFrame) -> DataFrame:
    line_data = power_flow_results.get(ComponentType.line, power_flow_results.get("line"))
    if line_data is None:
        raise TapOptimizationError("Line results not found in power flow output.")

    timestamps = pd.Index(active_load_profiles.index)
    if line_data.ndim == 1:
        line_data = line_data[np.newaxis, :]
    if line_data.shape[0] != len(timestamps):
        raise TapOptimizationError("Timestamp count does not match number of batch results.")

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
    return pd.DataFrame(
        {
            "Line_ID": line_ids,
            "Total_Loss": total_loss_kwh,
            "Max_Loading": max_loading,
            "Max_Loading_Timestamp": max_loading_ts,
            "Min_Loading": min_loading,
            "Min_Loading_Timestamp": min_loading_ts,
        }
    ).set_index("Line_ID")


def _get_component_from_dataset(dataset: Dataset, component_name: str) -> np.ndarray:
    for component_type, component in dataset.items():
        if str(component_type) == component_name:
            return component
    raise TapOptimizationError(f"Grid dataset does not contain a {component_name} component.")


class TapPositionOptimization:
    def __init__(
        self,
        dataset: Dataset,
        active_load_profiles: pd.DataFrame,
        reactive_load_profiles: pd.DataFrame,
        transformer_id: int | None = None,
    ) -> None:
        self._dataset = dataset
        self._active_load_profiles = active_load_profiles
        self._reactive_load_profiles = reactive_load_profiles
        self._tap_transformer_id = transformer_id

        self._validate_tap_inputs()
        self._tap_transformer_id = self._resolve_transformer_id(transformer_id)

    def optimize_tap_position(self, criterion: TapOptimizationCriterion, *args, **kwargs) -> TapOptimizationResult:
        if not callable(criterion):
            raise TapOptimizationError("Tap optimization criterion must be callable.")

        tap_positions = self._possible_tap_positions()
        rows = []

        max_workers = min(len(tap_positions), cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            evaluations = executor.map(
                _evaluate_tap_position_candidate,
                tap_positions,
                repeat(self._dataset),
                repeat(self._active_load_profiles),
                repeat(self._reactive_load_profiles),
                repeat(self._tap_transformer_id),
                repeat(args),
                repeat(kwargs),
            )

            for evaluation in evaluations:
                criterion_value = float(criterion(evaluation.timestamp_table, evaluation.line_table))
                rows.append(
                    {
                        "tap_position": evaluation.tap_position,
                        "criterion_value": criterion_value,
                        "total_loss_kwh": evaluation.total_loss_kwh,
                        "average_voltage_deviation_pu": evaluation.average_voltage_deviation_pu,
                    }
                )

        all_tap_results = pd.DataFrame(
            rows,
            columns=[
                "tap_position",
                "criterion_value",
                "total_loss_kwh",
                "average_voltage_deviation_pu",
            ],
        )
        best_idx = all_tap_results["criterion_value"].idxmin()
        best_row = all_tap_results.loc[best_idx]

        return TapOptimizationResult(
            tap_position=int(best_row["tap_position"]),
            criterion=criterion,
            criterion_value=float(best_row["criterion_value"]),
            total_loss_kwh=float(best_row["total_loss_kwh"]),
            average_voltage_deviation_pu=float(best_row["average_voltage_deviation_pu"]),
            all_tap_results=all_tap_results,
        )

    def _evaluate_tap_position(
        self,
        tap_position: int,
        power_flow_args: tuple[Any, ...],
        power_flow_kwargs: dict[str, Any],
    ) -> _TapPositionEvaluation:
        return _evaluate_tap_position_candidate(
            tap_position=tap_position,
            dataset=self._dataset,
            active_load_profiles=self._active_load_profiles,
            reactive_load_profiles=self._reactive_load_profiles,
            tap_transformer_id=self._tap_transformer_id,
            power_flow_args=power_flow_args,
            power_flow_kwargs=power_flow_kwargs,
        )

    def _validate_tap_inputs(self) -> None:
        self._get_component("transformer")
        sym_load = self._get_component("sym_load")

        if not self._active_load_profiles.index.equals(self._reactive_load_profiles.index):
            raise TapOptimizationError("Active and reactive load profiles have mismatched timestamps.")
        if not self._active_load_profiles.columns.equals(self._reactive_load_profiles.columns):
            raise TapOptimizationError("Active and reactive load profiles have mismatched Load IDs.")

        load_ids = {int(load_id) for load_id in sym_load["id"]}
        profile_ids = {int(load_id) for load_id in self._active_load_profiles.columns}
        missing_ids = sorted(profile_ids - load_ids)
        if missing_ids:
            raise TapOptimizationError(f"Load profile contains IDs that are not sym_load IDs: {missing_ids}.")

    def _resolve_transformer_id(self, transformer_id: int | None) -> int:
        transformer = self._get_component("transformer")
        transformer_ids = [int(transformer_id) for transformer_id in transformer["id"]]

        if transformer_id is None:
            if len(transformer_ids) != 1:
                raise TapOptimizationError("Tap optimization requires exactly one transformer or an explicit ID.")
            return transformer_ids[0]

        if int(transformer_id) not in transformer_ids:
            raise TapOptimizationError(f"Transformer ID {transformer_id} is not present in the grid dataset.")
        return int(transformer_id)

    def _possible_tap_positions(self) -> list[int]:
        transformer = self._tap_transformer_row()
        required_fields = {"tap_pos", "tap_min", "tap_max"}
        if transformer.dtype.names is None or not required_fields.issubset(transformer.dtype.names):
            raise TapOptimizationError("Transformer data does not expose tap position information.")

        tap_min = int(transformer["tap_min"])
        tap_max = int(transformer["tap_max"])

        if tap_min == np.iinfo(np.int8).min or tap_max == np.iinfo(np.int8).min:
            raise TapOptimizationError("Transformer tap_min and tap_max must both be defined.")
        if tap_min > tap_max:
            raise TapOptimizationError("Transformer tap_min cannot be greater than tap_max.")

        return list(range(tap_min, tap_max + 1))

    def _tap_transformer_row(self) -> np.void:
        transformer = self._get_component("transformer")
        matches = transformer["id"] == self._tap_transformer_id
        if not np.any(matches):
            raise TapOptimizationError(f"Transformer ID {self._tap_transformer_id} is not present in the dataset.")
        return transformer[matches][0]

    def _copy_dataset_with_tap_position(self, tap_position: int) -> Dataset:
        return _copy_dataset_with_tap_position(self._dataset, self._tap_transformer_id, tap_position)

    def _run_time_series_power_flow(self, dataset: Dataset, *args, **kwargs) -> dict:
        return _run_time_series_power_flow(
            dataset,
            self._active_load_profiles,
            self._reactive_load_profiles,
            *args,
            **kwargs,
        )

    def _create_pgm_batch_dataset(self) -> dict:
        return _create_pgm_batch_dataset(self._active_load_profiles, self._reactive_load_profiles)

    def _output_table_row_per_timestamp(self, power_flow_results: dict) -> DataFrame:
        return _output_table_row_per_timestamp(power_flow_results, self._active_load_profiles)

    def _output_table_row_per_line(self, power_flow_results: dict) -> DataFrame:
        return _output_table_row_per_line(power_flow_results, self._active_load_profiles)

    def _get_component(self, component_name: str) -> np.ndarray:
        return self._get_component_from_dataset(self._dataset, component_name)

    @staticmethod
    def _get_component_from_dataset(dataset: Dataset, component_name: str) -> np.ndarray:
        return _get_component_from_dataset(dataset, component_name)
