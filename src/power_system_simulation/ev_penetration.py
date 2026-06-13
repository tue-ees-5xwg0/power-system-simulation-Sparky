import math
from typing import Any

import numpy as np
import pandas as pd
from power_grid_model import PowerGridModel, power_grid_meta_data
from power_grid_model._core.data_types import Dataset

from power_system_simulation.validate import ValidationException


def run_ev_penetration(
    dataset: Dataset,
    active_load_profiles: pd.DataFrame,
    reactive_load_profiles: pd.DataFrame,
    ev_pool: pd.DataFrame,
    feeder_line_ids: list[int],
    graph_processor: Any,
    penetration_level: float,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(penetration_level, (int, float)):
        raise ValidationException("Penetration level must be a number.")
    if penetration_level <= 0.0 or penetration_level > 1.0:
        raise ValidationException(
            f"Penetration level must be in range (0.0, 1.0]. Got {penetration_level}."
        )

    total_houses = len(dataset["sym_load"]["id"])
    evs_per_feeder = math.floor(penetration_level * total_houses / len(feeder_line_ids))

    rng = np.random.default_rng(random_seed)
    active_load_profiles_with_ev = active_load_profiles.copy()
    available_ev_profiles = list(range(len(ev_pool.columns)))

    for feeder_line_id in feeder_line_ids:
        downstream_vertices = graph_processor.find_downstream_vertices(feeder_line_id)

        sym_load_data = dataset["sym_load"]
        downstream_load_ids: list[int] = []
        for node_id in downstream_vertices:
            matching_loads = sym_load_data[sym_load_data["node"] == node_id]["id"]
            downstream_load_ids.extend([int(load_id) for load_id in matching_loads])

        num_selected = min(evs_per_feeder, len(downstream_load_ids))
        if num_selected > 0:
            selected_load_ids = rng.choice(downstream_load_ids, size=num_selected, replace=False)

            for load_id in selected_load_ids:
                if not available_ev_profiles:
                    raise ValidationException(
                        "Not enough EV profiles to assign to all selected houses."
                    )

                profile_idx = rng.choice(len(available_ev_profiles))
                ev_profile_col = available_ev_profiles.pop(profile_idx)

                load_id_str = str(load_id)
                if load_id_str in active_load_profiles_with_ev.columns:
                    active_load_profiles_with_ev[load_id_str] += ev_pool.iloc[:, ev_profile_col]

    power_flow_results = _run_time_series_power_flow(
        dataset,
        active_load_profiles_with_ev,
        reactive_load_profiles,
    )

    timestamp_table = _output_table_row_per_timestamp(power_flow_results, active_load_profiles_with_ev)
    line_table = _output_table_row_per_line(power_flow_results, active_load_profiles_with_ev)

    return timestamp_table, line_table


def _run_time_series_power_flow(
    dataset: Dataset,
    active_load_profiles: pd.DataFrame,
    reactive_load_profiles: pd.DataFrame,
) -> dict:
    try:
        model = PowerGridModel(dataset)
        timestamps = active_load_profiles.index
        load_ids = list(active_load_profiles.columns)

        update_meta = power_grid_meta_data["update"]["sym_load"]
        sym_load_dtype = update_meta.dtype
        status_nan = update_meta.nan_scalar["status"][0]

        sym_load_updates = []
        for timestamp in timestamps:
            timestamp_updates = []
            for load_id in load_ids:
                timestamp_updates.append(
                    (
                        int(load_id),
                        status_nan,
                        float(active_load_profiles.loc[timestamp, load_id]),
                        float(reactive_load_profiles.loc[timestamp, load_id]),
                    )
                )
            sym_load_updates.append(np.array(timestamp_updates, dtype=sym_load_dtype))

        batch_dataset = {"sym_load": np.stack(sym_load_updates, axis=0)}

        return model.calculate_power_flow(
            update_data=batch_dataset,
            symmetric=True,
        )
    except Exception as e:
        raise ValidationException(f"Power flow failed: {e}") from e


def _output_table_row_per_timestamp(power_flow_results: dict, active_load_profiles: pd.DataFrame) -> pd.DataFrame:
    node_data = power_flow_results.get("node", power_flow_results.get("node"))
    if node_data is None:
        raise ValidationException("Node results not found in power flow output.")

    timestamps = pd.to_datetime(active_load_profiles.index)
    if node_data.ndim == 1:
        node_data = node_data[np.newaxis, :]
    if node_data.shape[0] != len(timestamps):
        raise ValidationException("Timestamp count does not match number of batch results.")

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


def _output_table_row_per_line(power_flow_results: dict, active_load_profiles: pd.DataFrame) -> pd.DataFrame:
    line_data = power_flow_results.get("line", power_flow_results.get("line"))
    if line_data is None:
        raise ValidationException("Line results not found in power flow output.")

    timestamps = pd.to_datetime(active_load_profiles.index)
    if line_data.ndim == 1:
        line_data = line_data[np.newaxis, :]
    if line_data.shape[0] != len(timestamps):
        raise ValidationException("Timestamp count does not match number of batch results.")

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
