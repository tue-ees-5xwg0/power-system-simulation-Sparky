import pandas as pd

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
    active_profiles = pd.DataFrame({"node": [1, 2, 3], "active_load": [100, 200, 300]})
    reactive_profiles = pd.DataFrame({"node": [1, 2, 3], "reactive_load": [50, 100, 150]})
    try:
        _validate_profiles_match(active_profiles, reactive_profiles)
    except ValidationException:
        raise AssertionError("ValidationException was raised for matching profiles.") from None

    # Test with non-matching profiles
    reactive_profiles_non_matching = pd.DataFrame({"node": [1, 2], "reactive_load": [50, 100]})
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

    pd.testing.assert_frame_equal(output_row_per_line, expected_row_per_line, check_dtype=False, check_index_type=False, rtol=ASSERT_MAX_RTOLERANCE, atol=ASSERT_MAX_ATOLERANCE)
    pd.testing.assert_frame_equal(output_row_per_timestamp, expected_row_per_timestamp, check_dtype=False, check_index_type=False, rtol=ASSERT_MAX_RTOLERANCE, atol=ASSERT_MAX_ATOLERANCE)