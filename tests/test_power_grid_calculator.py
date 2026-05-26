import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from PowerGridModel.power_grid_calculator import (
    GridModel,
    ProfilesNotMatchingError,
    ValidationException,
    _validate_active_reactive_profiles,
    _validate_load_profile,
    _validate_power_grid_model,
    _validate_profiles_match,
)

FILE_PATH_VALID_INPUT = "tests/PGM_TestData/input"
FILE_PATH_FALSE_INPUT = "tests/PGM_TestData/false_input"
FILE_PATH_EXPECTED_OUTPUT = "tests/PGM_TestData/expected_output"

ASSERT_MAX_RTOLERANCE = 1e-6
ASSERT_MAX_ATOLERANCE = 1e-6

PD_ASSERT_FRAME_EQUAL_KWARGS = {
    "check_dtype": False,
    "check_index_type": False,
    "rtol": ASSERT_MAX_RTOLERANCE,
    "atol": ASSERT_MAX_ATOLERANCE,
}


def test_validate_power_grid_model():
    # Test with valid input
    try:
        _validate_power_grid_model(FILE_PATH_VALID_INPUT + "/input_network_data.json")
    except ValidationException:
        raise AssertionError("ValidationException was raised for valid input.") from None

    # Test with empty string
    try:
        _validate_power_grid_model("")
        raise AssertionError("ValidationException was not raised for empty string.")
    except ValidationException:
        pass

    # Test with invalid file path
    try:
        _validate_power_grid_model(FILE_PATH_FALSE_INPUT + "/invalid_path.json")
        raise AssertionError("ValidationException was not raised for invalid file path.")
    except ValidationException:
        pass

    # Test with invalid file extension
    try:
        _validate_power_grid_model(FILE_PATH_FALSE_INPUT + "/wrong.extension")
        raise AssertionError("ValidationException was not raised for invalid file extension.")
    except ValidationException:
        pass

    # Test with inconsistent data
    try:
        _validate_power_grid_model(FILE_PATH_FALSE_INPUT + "/input_network_data.json")
        raise AssertionError("ValidationException was not raised for inconsistent data.")
    except ValidationException:
        pass


def test_validate_load_profiles():
    # Test with valid input
    try:
        _validate_active_reactive_profiles(
            FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
            FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        )
    except ValidationException:
        raise AssertionError("ValidationException was raised for valid input.") from None

    # Test with invalid file path
    try:
        _validate_load_profile("Test", FILE_PATH_FALSE_INPUT + "/invalid_active_path.parquet")
        raise AssertionError("ValidationException was not raised for invalid file path.")
    except ValidationException:
        pass

    # Test with invalid file extension
    try:
        _validate_load_profile("Test", FILE_PATH_FALSE_INPUT + "/wrong.extension")
        raise AssertionError("ValidationException was not raised for invalid file extension.")
    except ValidationException:
        pass

    # Test with empty string
    try:
        _validate_load_profile("Test", "")
        raise AssertionError("ValidationException was not raised for empty string.")
    except ValidationException:
        pass


def test_validate_profiles_match():
    # Test with matching profiles
    active_profiles = pd.DataFrame({1: [100, 200, 300], 2: [100, 200, 300]})
    reactive_profiles = pd.DataFrame({1: [50, 100, 150], 2: [50, 100, 150]})
    try:
        _validate_profiles_match(active_profiles, reactive_profiles)
    except ValidationException:
        raise AssertionError("ValidationException was raised for matching profiles.") from None

    # Test with non-matching profiles
    reactive_profiles_non_matching = pd.DataFrame({1: [50, 100, 150], 3: [50, 100, 150]})
    try:
        _validate_profiles_match(active_profiles, reactive_profiles_non_matching)
        raise AssertionError("ValidationException was not raised for non-matching profiles.")
    except ProfilesNotMatchingError:
        pass


def test_valid_model_and_profiles():
    # Test with valid model and profiles
    try:
        GridModel(
            power_grid_model_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
            active_load_profiles_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
            reactive_load_profiles_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        )
    except ValidationException:
        raise AssertionError("ValidationException was raised for valid model and profiles.") from None


def test_expected_output():
    # Test that the output matches expected output
    model = GridModel(
        power_grid_model_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profiles_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profiles_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
    )

    output_row_per_timestamp, output_row_per_line = model.AggregateResults()

    expected_row_per_line = pd.read_parquet(FILE_PATH_EXPECTED_OUTPUT + "/output_table_row_per_line.parquet")
    expected_row_per_timestamp = pd.read_parquet(FILE_PATH_EXPECTED_OUTPUT + "/output_table_row_per_timestamp.parquet")

    pd.testing.assert_frame_equal(output_row_per_line, expected_row_per_line, **PD_ASSERT_FRAME_EQUAL_KWARGS)
    pd.testing.assert_frame_equal(output_row_per_timestamp, expected_row_per_timestamp, **PD_ASSERT_FRAME_EQUAL_KWARGS)


# Test case to validate profile ID mismatch
def test_validate_profiles_mismatch_load_ids():
    active_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 2: [100.0, 220.0]}, index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:15"])
    )
    reactive_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 3: [100.0, 220.0]}, index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:15"])
    )

    try:
        _validate_profiles_match(active_profiles, reactive_profiles)
        raise AssertionError("ProfilesNotMatchingError was not raised for mismatched load IDs.")
    except ProfilesNotMatchingError:
        pass


# Testcase to validate profile timestamp mismatch
def test_validate_profiles_mismatch_timestamp():
    active_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 2: [100.0, 220.0]}, index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:15"])
    )
    reactive_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 2: [100.0, 220.0]}, index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:30"])
    )

    try:
        _validate_profiles_match(active_profiles, reactive_profiles)
        raise AssertionError("ProfilesNotMatchingError was not raised for mismatched timestamps.")
    except ProfilesNotMatchingError:
        pass


# Test case to run a valid and invalid batch run
def test_valid_batch_run():
    model = GridModel(
        power_grid_model_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profiles_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profiles_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
    )

    try:
        model.AggregateResults()
    except Exception as e:
        raise AssertionError(f"Valid batch run crashed unexpectedly during execution: {e}") from None


def test_invalid_batch_run():
    model = GridModel(
        power_grid_model_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profiles_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profiles_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
    )

    model._active_load_profiles = model._active_load_profiles * 1e9
    model._pgm_batch_dataset = model._create_pgm_batch_dataset()

    try:
        model.AggregateResults()
        raise AssertionError("Invalid batch run did not raise an exception as expected.")
    except ValidationException:
        pass


def test_full_system_accuracy_againts_expected_results():
    model = GridModel(
        power_grid_model_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profiles_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profiles_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
    )

    calculated_timestamp_table, calculated_line_table = model.AggregateResults()

    expected_timestamp_table = pd.read_parquet(FILE_PATH_EXPECTED_OUTPUT + "/output_table_row_per_timestamp.parquet")
    expected_line_table = pd.read_parquet(FILE_PATH_EXPECTED_OUTPUT + "/output_table_row_per_line.parquet")

    try:
        assert_frame_equal(
            calculated_timestamp_table, expected_timestamp_table, check_exact=False, rtol=1e-5, check_dtype=False
        )

    except AssertionError as e:
        raise AssertionError(
            f"The calculated Timestamp (Node) Table does not match the expected output!\nDetails: {e}"
        ) from None

    try:
        assert_frame_equal(
            calculated_line_table,
            expected_line_table,
            check_exact=False,
            rtol=1e-5,
            check_dtype=False,
            check_index_type=False,
        )
    except AssertionError as e:
        raise AssertionError(f"The calculated Line Table does not match the expected output!\nDetails: {e}") from None


def test_output_node_table():
    model = GridModel(
        power_grid_model_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profiles_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profiles_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
    )

    #Test empty node table
    try:
        model._output_table_row_per_timestamp({})
        raise AssertionError(
            "Expected ValidationException was not raised when outputting node table with empty line table."
        )
    except ValueError:
        pass

    #Test node table with invalid node data
    bad_node_data = {"node": np.zeros(99)}

    try:
        model._output_table_row_per_timestamp(bad_node_data)
        raise AssertionError(
            "Expected ValidationException was not raised when outputting node table with invalid node data."
        )
    except ValueError:
        pass

    #Test node table with wrong shape
    mock_bad_shape_data = {"node": np.zeros((99, 1))}

    try:
        model._output_table_row_per_timestamp(mock_bad_shape_data)
        raise AssertionError(
            "Expected ValidationException was not raised when outputting node table with invalid node data."
        )
    except ValueError:
        pass

    #Test node table with NaN values
    timestamps_len = len(model._active_load_profiles.index)
    num_nodes = 5
    mock_nan_node_array = np.zeros((timestamps_len, num_nodes), dtype=[('id', 'i4'), ('u_pu', 'f8')])
    mock_nan_node_array['id'] = np.arange(num_nodes)
    mock_nan_node_array['u_pu'] = np.nan
    mock_nan_data = {"node": mock_nan_node_array}

    result = model._output_table_row_per_timestamp(mock_nan_data)
    assert result["Max_Voltage"].isna().all(), "Expected all Max_Voltage values to be NaN"
    assert result["Min_Voltage"].isna().all(), "Expected all Min_Voltage values to be NaN"




def test_output_line_table():
    model = GridModel(
        power_grid_model_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        active_load_profiles_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profiles_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
    )

    #Test empty line table
    try:
        model._output_table_row_per_line({})
        raise AssertionError(
            "Expected ValidationException was not raised when outputting line table with empty line table."
        )
    except ValueError:
        pass

    bad_line_data = {"line": np.zeros(99)}

    #Test line table with invalid line data
    try:
        model._output_table_row_per_line(bad_line_data)
        raise AssertionError(
            "Expected ValidationException was not raised when outputting line table with invalid line data."
        )
    except ValueError:
        pass

    #Test line table with wrong shape
    mock_bad_shape_data = {"line": np.zeros((99, 1))}

    try:
        model._output_table_row_per_line(mock_bad_shape_data)
        raise AssertionError(
            "Expected ValidationException was not raised when outputting line table with invalid line data."
        )
    except ValueError:
        pass

    #Test line table with NaN values
    timestamps_len = len(model._active_load_profiles.index)
    num_lines = 5
    mock_nan_line_array = np.zeros((timestamps_len, num_lines),
                                   dtype=[('id', 'i4'), ('p_from', 'f8'), ('p_to', 'f8'), ('loading', 'f8')])
    mock_nan_line_array['id'] = np.arange(num_lines)
    mock_nan_line_array['p_from'] = np.nan
    mock_nan_line_array['p_to'] = np.nan
    mock_nan_line_array['loading'] = np.nan
    mock_nan_line_data = {"line": mock_nan_line_array}

    result = model._output_table_row_per_line(mock_nan_line_data)
    assert result["Max_Loading"].isna().all(), "Expected all Max_Loading values to be NaN"
    assert result["Min_Loading"].isna().all(), "Expected all Min_Loading values to be NaN"
