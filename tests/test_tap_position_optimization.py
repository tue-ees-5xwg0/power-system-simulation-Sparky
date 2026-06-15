import numpy as np
import pandas as pd
import pytest
from power_grid_model import BranchSide, LoadGenType, WindingType, initialize_array

from power_system_simulation.lv_grid_analytics import LVGridAnalytics
from power_system_simulation.tap_position_optimization import (
    TapOptimizationError,
    TapOptimizationResult,
    TapPositionOptimization,
    minimize_average_voltage_deviation,
    minimize_total_loss,
)


def _transformer_dataset(tap_min: int = -1, tap_max: int = 1) -> dict:
    node = initialize_array("input", "node", 3)
    node["id"] = [1, 2, 3]
    node["u_rated"] = [10500.0, 400.0, 400.0]

    source = initialize_array("input", "source", 1)
    source["id"] = [10]
    source["node"] = [1]
    source["status"] = [1]
    source["u_ref"] = [1.0]
    source["sk"] = [1e9]

    transformer = initialize_array("input", "transformer", 1)
    transformer["id"] = [20]
    transformer["from_node"] = [1]
    transformer["to_node"] = [2]
    transformer["from_status"] = [1]
    transformer["to_status"] = [1]
    transformer["u1"] = [10500.0]
    transformer["u2"] = [400.0]
    transformer["sn"] = [400000.0]
    transformer["uk"] = [4.0]
    transformer["pk"] = [4000.0]
    transformer["i0"] = [0.1]
    transformer["p0"] = [400.0]
    transformer["winding_from"] = [WindingType.delta.value]
    transformer["winding_to"] = [WindingType.wye_n.value]
    transformer["clock"] = [5]
    transformer["tap_side"] = [BranchSide.to_side.value]
    transformer["tap_pos"] = [0]
    transformer["tap_min"] = [tap_min]
    transformer["tap_max"] = [tap_max]
    transformer["tap_nom"] = [0]
    transformer["tap_size"] = [2.5]

    line = initialize_array("input", "line", 1)
    line["id"] = [30]
    line["from_node"] = [2]
    line["to_node"] = [3]
    line["from_status"] = [1]
    line["to_status"] = [1]
    line["r1"] = [0.01]
    line["x1"] = [0.01]
    line["c1"] = [0.0]
    line["tan1"] = [0.0]
    line["i_n"] = [500.0]

    load = initialize_array("input", "sym_load", 1)
    load["id"] = [40]
    load["node"] = [3]
    load["status"] = [1]
    load["type"] = [LoadGenType.const_power.value]
    load["p_specified"] = [0.0]
    load["q_specified"] = [0.0]

    return {
        "node": node,
        "source": source,
        "transformer": transformer,
        "line": line,
        "sym_load": load,
    }


def _profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    active = pd.DataFrame({40: [10000.0, 11000.0, 12000.0]}, index=index)
    reactive = pd.DataFrame({40: [1000.0, 1200.0, 1400.0]}, index=index)
    return active, reactive


def test_minimize_total_loss_returns_tap_result():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(), active, reactive)

    result = optimizer.optimize_tap_position(minimize_total_loss)

    assert isinstance(result, TapOptimizationResult)
    assert result.tap_position in {-1, 0, 1}
    assert result.criterion is minimize_total_loss
    assert result.criterion_value == pytest.approx(result.total_loss_kwh)
    assert list(result.all_tap_results.columns) == [
        "tap_position",
        "criterion_value",
        "total_loss_kwh",
        "average_voltage_deviation_pu",
    ]
    assert len(result.all_tap_results) == 3


def test_minimize_average_voltage_deviation_returns_tap_result():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(), active, reactive)

    result = optimizer.optimize_tap_position(minimize_average_voltage_deviation)

    assert isinstance(result, TapOptimizationResult)
    assert result.tap_position in {-1, 0, 1}
    assert result.criterion_value == pytest.approx(result.average_voltage_deviation_pu)


def test_custom_criterion_receives_aggregation_tables_and_controls_selection():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(), active, reactive)
    calls = []

    def custom_criterion(timestamp_table: pd.DataFrame, line_table: pd.DataFrame) -> float:
        calls.append((timestamp_table, line_table))
        assert "Max_Voltage" in timestamp_table.columns
        assert "Total_Loss" in line_table.columns
        return float(len(calls))

    result = optimizer.optimize_tap_position(custom_criterion)

    assert len(calls) == 3
    assert result.tap_position == -1
    assert result.criterion is custom_criterion


def test_missing_transformer_raises_tap_optimization_error():
    dataset = _transformer_dataset()
    dataset.pop("transformer")
    active, reactive = _profiles()

    with pytest.raises(TapOptimizationError):
        TapPositionOptimization(dataset, active, reactive)


def test_invalid_tap_bounds_raise_tap_optimization_error():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(tap_min=2, tap_max=1), active, reactive)

    with pytest.raises(TapOptimizationError):
        optimizer.optimize_tap_position(minimize_total_loss)


def test_lv_grid_analytics_wraps_inherited_tap_optimization():
    assert issubclass(LVGridAnalytics, TapPositionOptimization)
    assert LVGridAnalytics.optimize_tap_position is not TapPositionOptimization.optimize_tap_position


def test_invalid_criterion_raises_tap_optimization_error():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(), active, reactive)

    with pytest.raises(TapOptimizationError):
        optimizer.optimize_tap_position("not callable")


def test_profile_validation_errors_raise_tap_optimization_error():
    active, reactive = _profiles()
    bad_index = reactive.copy()
    bad_index.index = pd.date_range("2024-01-02", periods=3, freq="h")

    with pytest.raises(TapOptimizationError):
        TapPositionOptimization(_transformer_dataset(), active, bad_index)

    bad_columns = reactive.rename(columns={40: 41})
    with pytest.raises(TapOptimizationError):
        TapPositionOptimization(_transformer_dataset(), active, bad_columns)

    unknown_load = active.rename(columns={40: 999})
    with pytest.raises(TapOptimizationError):
        TapPositionOptimization(_transformer_dataset(), unknown_load, unknown_load)


def test_transformer_id_validation_errors_raise_tap_optimization_error():
    active, reactive = _profiles()
    dataset = _transformer_dataset()
    dataset["transformer"] = np.concatenate([dataset["transformer"], dataset["transformer"].copy()])
    dataset["transformer"]["id"][1] = 21

    with pytest.raises(TapOptimizationError):
        TapPositionOptimization(dataset, active, reactive)

    with pytest.raises(TapOptimizationError):
        TapPositionOptimization(_transformer_dataset(), active, reactive, transformer_id=999)

    optimizer = TapPositionOptimization(dataset, active, reactive, transformer_id=21)
    assert optimizer._tap_transformer_id == 21


def test_missing_tap_fields_and_missing_tap_bounds_raise_tap_optimization_error():
    active, reactive = _profiles()
    dataset = _transformer_dataset()
    dataset["transformer"] = np.array([(20,)], dtype=[("id", "i4")])
    optimizer = TapPositionOptimization(dataset, active, reactive)

    with pytest.raises(TapOptimizationError):
        optimizer.optimize_tap_position(minimize_total_loss)

    dataset = _transformer_dataset()
    dataset["transformer"]["tap_min"] = [np.iinfo(np.int8).min]
    optimizer = TapPositionOptimization(dataset, active, reactive)

    with pytest.raises(TapOptimizationError):
        optimizer.optimize_tap_position(minimize_total_loss)


def test_missing_transformer_row_after_init_raises_tap_optimization_error():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(), active, reactive)
    optimizer._tap_transformer_id = 999

    with pytest.raises(TapOptimizationError):
        optimizer.optimize_tap_position(minimize_total_loss)


def test_aggregation_helpers_handle_single_result_and_missing_tables():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(), active.iloc[:1], reactive.iloc[:1])

    node_data = np.zeros(2, dtype=[("id", "i4"), ("u_pu", "f8")])
    node_data["id"] = [1, 2]
    node_data["u_pu"] = np.nan
    timestamp_table = optimizer._output_table_row_per_timestamp({"node": node_data})
    assert timestamp_table["Max_Voltage"].isna().all()

    line_data = np.zeros(
        1,
        dtype=[("id", "i4"), ("p_from", "f8"), ("p_to", "f8"), ("loading", "f8")],
    )
    line_data["id"] = [30]
    line_data["p_from"] = [1.0]
    line_data["p_to"] = [1.0]
    line_data["loading"] = np.nan
    line_table = optimizer._output_table_row_per_line({"line": line_data})
    assert line_table["Total_Loss"].iloc[0] == 0.0
    assert line_table["Max_Loading"].isna().all()

    with pytest.raises(TapOptimizationError):
        optimizer._output_table_row_per_timestamp({})
    with pytest.raises(TapOptimizationError):
        optimizer._output_table_row_per_line({})


def test_aggregation_helpers_raise_on_wrong_batch_length():
    active, reactive = _profiles()
    optimizer = TapPositionOptimization(_transformer_dataset(), active, reactive)

    node_data = np.zeros((1, 1), dtype=[("id", "i4"), ("u_pu", "f8")])
    line_data = np.zeros(
        (1, 1),
        dtype=[("id", "i4"), ("p_from", "f8"), ("p_to", "f8"), ("loading", "f8")],
    )

    with pytest.raises(TapOptimizationError):
        optimizer._output_table_row_per_timestamp({"node": node_data})
    with pytest.raises(TapOptimizationError):
        optimizer._output_table_row_per_line({"line": line_data})
