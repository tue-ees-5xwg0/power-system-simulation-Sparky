import pytest

from power_system_simulation.lv_grid_analytics import (
    LVGridAnalytics,
)
from power_system_simulation.tap_position_optimization import minimize_average_voltage_deviation, minimize_total_loss

FILE_PATH_VALID_INPUT = "tests/small_network"
FILE_PATH_FALSE_INPUT = "tests/small_network"
FILE_PATH_EXPECTED_OUTPUT = "tests/small_network"

ASSERT_MAX_RTOLERANCE = 1e-6
ASSERT_MAX_ATOLERANCE = 1e-6

PD_ASSERT_FRAME_EQUAL_KWARGS = {
    "check_dtype": False,
    "check_index_type": False,
    "rtol": ASSERT_MAX_RTOLERANCE,
    "atol": ASSERT_MAX_ATOLERANCE,
}


@pytest.fixture
def valid_grid() -> LVGridAnalytics:
    return LVGridAnalytics(
        grid_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profile_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profile_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        ev_profile_path=FILE_PATH_VALID_INPUT + "/ev_active_power_profile.parquet",
        meta_data=FILE_PATH_VALID_INPUT + "/meta_data.json",
    )


def test_tap_optimization_inside_lv_grid_analytics_minimize_average_voltage_deviation(valid_grid):
    # This test ensures that the optimize_tap_position method is available and can be called without errors.
    try:
        # Using the minimize_average_voltage_deviation criterion
        valid_grid.optimize_tap_position(minimize_average_voltage_deviation)
    except Exception as e:
        pytest.fail(f"optimize_tap_position raised an exception: {e}")


def test_tap_optimization_inside_lv_grid_analytics_minimize_total_loss(valid_grid):
    # This test ensures that the optimize_tap_position method is available and can be called without errors.
    try:
        valid_grid.optimize_tap_position(minimize_total_loss)  # Using the minimize_total_loss criterion
    except Exception as e:
        pytest.fail(f"optimize_tap_position raised an exception: {e}")
