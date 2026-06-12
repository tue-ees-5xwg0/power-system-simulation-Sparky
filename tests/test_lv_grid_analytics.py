import numpy as np
import pytest

from power_system_simulation.validate import ProfilesNotMatchingError, ValidationException
from power_system_simulation.lv_grid_analytics import (
    Assignment3ValidationError,
    LVGridAnalytics,
    ProfileMismatchError,
)

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
        feeder_line_ids=[16, 20],  # Replace with actual valid feeder IDs for your test network if needed
        active_load_profile_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profile_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        ev_profile_path=FILE_PATH_VALID_INPUT + "/ev_active_power_profile.parquet",
    )


def test_power_grid_initialization(valid_grid):
    # Initialization logic is implicitly tested via the fixture.
    # This test ensures it does not raise exceptions.
    assert valid_grid is not None


def test_initialization_accepts_metadata_path():
    grid = LVGridAnalytics(
        grid_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profile_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profile_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        ev_profile_path=FILE_PATH_VALID_INPUT + "/ev_active_power_profile.parquet",
        meta_data=FILE_PATH_VALID_INPUT + "/meta_data.json",
    )

    assert grid._feeder_line_ids == [16, 20]


def test_resolve_feeder_line_ids_accepts_metadata_path():
    feeder_line_ids = LVGridAnalytics._resolve_feeder_line_ids(None, FILE_PATH_VALID_INPUT + "/meta_data.json")

    assert feeder_line_ids == [16, 20]


def test_resolve_feeder_line_ids_rejects_both_feeder_sources():
    with pytest.raises(ValidationException, match="either feeder_line_ids or meta_data"):
        LVGridAnalytics._resolve_feeder_line_ids([16, 20], FILE_PATH_VALID_INPUT + "/meta_data.json")


def test_resolve_feeder_line_ids_rejects_missing_metadata():
    with pytest.raises(ValidationException, match="Metadata file not found"):
        LVGridAnalytics._resolve_feeder_line_ids(None, FILE_PATH_VALID_INPUT + "/missing_meta_data.json")


def test_resolve_feeder_line_ids_rejects_invalid_metadata(tmp_path):
    metadata_path = tmp_path / "meta_data.json"
    metadata_path.write_text('{"lv_feeders": ["16"]}')

    with pytest.raises(ValidationException, match="'lv_feeders' list of line IDs"):
        LVGridAnalytics._resolve_feeder_line_ids(None, metadata_path)


def test_validate_inputs_with_valid_data(valid_grid):
    try:
        valid_grid.validate_inputs()
    except Exception as e:
        pytest.fail(f"validate_inputs failed with valid data: {e}")


def test_validate_ev_profile_different_time_indices(valid_grid):
    # Remove the last row to cause a mismatch in the indices
    valid_grid._ev_pool = valid_grid._ev_pool.iloc[:-1]
    with pytest.raises(ProfilesNotMatchingError, match="different time indices"):
        valid_grid._validate_ev_profile()


def test_validate_ev_profile_fewer_columns(valid_grid):
    # Keep fewer EV profile columns than active load columns
    num_cols = max(1, len(valid_grid._active_load_profiles.columns) - 1)
    valid_grid._ev_pool = valid_grid._ev_pool.iloc[:, :num_cols]
    with pytest.raises(ProfilesNotMatchingError, match="fewer columns"):
        valid_grid._validate_ev_profile()


def test_validate_profile_sym_loads_mismatch(valid_grid):
    # Add a dummy ID that does not exist in the grid
    valid_grid._active_load_profiles[999999] = 0.0
    with pytest.raises(ProfileMismatchError, match="Load profile IDs do not perfectly match"):
        valid_grid._validate_profile_sym_loads()


def test_validate_transformer_count(valid_grid):
    # Add a second transformer
    valid_grid._dataset["transformer"] = np.repeat(valid_grid._dataset["transformer"], 2)
    with pytest.raises(ValidationException, match="one transformer"):
        valid_grid._validate_transformer()


def test_validate_source_count(valid_grid):
    # Add a second source
    valid_grid._dataset["source"] = np.repeat(valid_grid._dataset["source"], 2)
    with pytest.raises(ValidationException, match="exactly one source"):
        valid_grid._validate_source()


def test_validate_feeder_line_ids_invalid(valid_grid):
    # Set a feeder line ID that isn't in the dataset
    valid_grid._feeder_line_ids = [999999]
    with pytest.raises(ValidationException, match="not valid"):
        valid_grid._validate_feeder_line_ids()


def test_validate_feeder_connections_invalid(valid_grid):
    # Change transformer to_node to something that doesn't match the feeder's from_node
    valid_grid._dataset["transformer"]["to_node"][0] = 999999
    with pytest.raises(ValidationException, match="not connected to the transformer"):
        valid_grid._validate_feeder_connections()


def test_validate_topology_disconnected(valid_grid):
    # Disconnect the first enabled line to create a disjoint graph
    # This will trigger GraphNotFullyConnectedError wrapped into Assignment3ValidationError
    valid_grid._dataset["line"]["from_status"][0] = 0
    valid_grid._dataset["line"]["to_status"][0] = 0
    with pytest.raises(Assignment3ValidationError, match="not fully connected"):
        valid_grid._validate_topology()


def test_validate_topology_cyclic(valid_grid):
    # Enable all lines to create cycles in the ring-structured grid
    # This will trigger GraphCycleError wrapped into Assignment3ValidationError
    valid_grid._dataset["line"]["from_status"][:] = 1
    valid_grid._dataset["line"]["to_status"][:] = 1
    with pytest.raises(Assignment3ValidationError, match="contains cycles"):
        valid_grid._validate_topology()
