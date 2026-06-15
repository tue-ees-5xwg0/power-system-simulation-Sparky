import pandas as pd
import pytest

from power_system_simulation import ev_penetration as ev_penetration_module
from power_system_simulation.lv_grid_analytics import Assignment3ValidationError, LVGridAnalytics
from power_system_simulation.validate import ValidationException

FILE_PATH_VALID_INPUT = "tests/small_network"

PD_ASSERT_FRAME_EQUAL_KWARGS = {
    "check_dtype": False,
    "check_index_type": False,
    "rtol": 1e-6,
    "atol": 1e-6,
}


@pytest.fixture
def valid_grid() -> LVGridAnalytics:
    return LVGridAnalytics(
        grid_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        feeder_line_ids=[16, 20],
        active_load_profile_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profile_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        ev_profile_path=FILE_PATH_VALID_INPUT + "/ev_active_power_profile.parquet",
    )


def test_run_ev_penetration_with_valid_penetration_level(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.2,
        random_seed=42,
    )

    assert isinstance(timestamp_table, pd.DataFrame)
    assert isinstance(line_table, pd.DataFrame)
    assert len(timestamp_table) == len(valid_grid._active_load_profiles)

    expected_ts_columns = {"Max_Voltage", "Max_Voltage_Node", "Min_Voltage", "Min_Voltage_Node"}
    assert set(timestamp_table.columns) == expected_ts_columns

    expected_line_columns = {
        "Total_Loss",
        "Max_Loading",
        "Max_Loading_Timestamp",
        "Min_Loading",
        "Min_Loading_Timestamp",
    }
    assert set(line_table.columns) == expected_line_columns


def test_run_ev_penetration_adds_ev_profiles_to_integer_load_columns(valid_grid, monkeypatch):
    captured = {}

    def stop_after_assignment(dataset, active_load_profiles, reactive_load_profiles):
        captured["active_load_profiles"] = active_load_profiles.copy()
        raise ValidationException("stop after EV assignment")

    monkeypatch.setattr(ev_penetration_module, "_run_time_series_power_flow", stop_after_assignment)

    with pytest.raises(Assignment3ValidationError, match="stop after EV assignment"):
        valid_grid.run_ev_penetration(
            penetration_level=1.0,
            random_seed=42,
        )

    profile_delta = (captured["active_load_profiles"] - valid_grid._active_load_profiles).abs().sum().sum()
    assert profile_delta > 0.0


def test_run_ev_penetration_adds_ev_profiles_to_integer_load_columns(valid_grid, monkeypatch):
    captured = {}

    def stop_after_assignment(dataset, active_load_profiles, reactive_load_profiles):
        captured["active_load_profiles"] = active_load_profiles.copy()
        raise ValidationException("stop after EV assignment")

    monkeypatch.setattr(ev_penetration_module, "_run_time_series_power_flow", stop_after_assignment)

    with pytest.raises(Assignment3ValidationError, match="stop after EV assignment"):
        valid_grid.run_ev_penetration(
            penetration_level=1.0,
            random_seed=42,
        )

    profile_delta = (captured["active_load_profiles"] - valid_grid._active_load_profiles).abs().sum().sum()
    assert profile_delta > 0.0


def test_run_ev_penetration_accepts_penetration_level_zero(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.0,
        random_seed=42,
    )

    assert isinstance(timestamp_table, pd.DataFrame)
    assert isinstance(line_table, pd.DataFrame)
    assert len(timestamp_table) == len(valid_grid._active_load_profiles)


def test_run_ev_penetration_rejects_penetration_level_greater_than_one(valid_grid):
    with pytest.raises(Assignment3ValidationError, match="must be in range \\[0.0, 1.0\\]"):
        valid_grid.run_ev_penetration(penetration_level=1.5)


def test_run_ev_penetration_rejects_negative_penetration_level(valid_grid):
    with pytest.raises(Assignment3ValidationError, match="must be in range \\[0.0, 1.0\\]"):
        valid_grid.run_ev_penetration(penetration_level=-0.1)


def test_run_ev_penetration_rejects_non_numeric_penetration_level(valid_grid):
    with pytest.raises(Assignment3ValidationError, match="must be a number"):
        valid_grid.run_ev_penetration(penetration_level="0.2")


def test_run_ev_penetration_random_seed_reproducibility(valid_grid):
    ts1, lines1 = valid_grid.run_ev_penetration(
        penetration_level=0.3,
        random_seed=12345,
    )
    ts2, lines2 = valid_grid.run_ev_penetration(
        penetration_level=0.3,
        random_seed=12345,
    )

    pd.testing.assert_frame_equal(ts1, ts2, **PD_ASSERT_FRAME_EQUAL_KWARGS)
    pd.testing.assert_frame_equal(lines1, lines2, **PD_ASSERT_FRAME_EQUAL_KWARGS)


def test_run_ev_penetration_different_seeds_are_handled(valid_grid):
    ts1, lines1 = valid_grid.run_ev_penetration(
        penetration_level=0.5,
        random_seed=111,
    )
    ts2, lines2 = valid_grid.run_ev_penetration(
        penetration_level=0.5,
        random_seed=222,
    )

    assert isinstance(ts1, pd.DataFrame)
    assert isinstance(lines1, pd.DataFrame)
    assert isinstance(ts2, pd.DataFrame)
    assert isinstance(lines2, pd.DataFrame)


def test_run_ev_penetration_max_penetration_level(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=1.0,
        random_seed=42,
    )

    assert isinstance(timestamp_table, pd.DataFrame)
    assert isinstance(line_table, pd.DataFrame)
    assert len(timestamp_table) == len(valid_grid._active_load_profiles)


def test_run_ev_penetration_min_penetration_level(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.01,
        random_seed=42,
    )

    assert isinstance(timestamp_table, pd.DataFrame)
    assert isinstance(line_table, pd.DataFrame)


def test_run_ev_penetration_none_random_seed(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.2,
        random_seed=None,
    )

    assert isinstance(timestamp_table, pd.DataFrame)
    assert isinstance(line_table, pd.DataFrame)


def test_run_ev_penetration_wraps_validation_errors(valid_grid):
    valid_grid._feeder_line_ids = [999999]
    with pytest.raises(Assignment3ValidationError, match="Input validation failed"):
        valid_grid.run_ev_penetration(penetration_level=0.2)


def test_run_ev_penetration_returns_correct_index_types(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.2,
        random_seed=42,
    )

    assert isinstance(timestamp_table.index, (pd.DatetimeIndex, pd.Index))
    assert line_table.index.name == "Line_ID"


def test_run_ev_penetration_voltage_values_in_valid_range(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.15,
        random_seed=42,
    )

    assert (timestamp_table["Max_Voltage"] > 0).all()
    assert (timestamp_table["Min_Voltage"] > 0).all()
    assert (timestamp_table["Max_Voltage"] <= 2.0).all()
    assert (timestamp_table["Min_Voltage"] >= 0.5).all()


def test_run_ev_penetration_loading_values_valid(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.1,
        random_seed=42,
    )

    valid_loading = line_table["Max_Loading"].dropna()
    if len(valid_loading) > 0:
        assert (valid_loading >= 0).all()
        assert (valid_loading <= 1).all()


def test_run_ev_penetration_loss_values_non_negative(valid_grid):
    timestamp_table, line_table = valid_grid.run_ev_penetration(
        penetration_level=0.2,
        random_seed=42,
    )

    assert (line_table["Total_Loss"] >= 0).all()
