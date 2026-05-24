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

#Test case to validate profile ID mismatch
def test_validate_profiles_mismatch_load_ids():
    active_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 2: [100.0, 220.0]},
        index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:15"])
    )
    reactive_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 3: [100.0, 220.0]},
        index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:15"])
    )

    try:
        _validate_profiles_match(active_profiles, reactive_profiles)
        raise AssertionError("ProfilesNotMatchingError was not raised for mismatched load IDs.")
    except ProfilesNotMatchingError:
        pass

#Testcase to validate profile timestamp mismatch
def test_validate_profiles_mismatch_timestamp():
    active_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 2: [100.0, 220.0]},
        index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:15"])
    )
    reactive_profiles = pd.DataFrame(
        {1: [100.0, 200.0], 2: [100.0, 220.0]},
        index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:30"])
    )

    try:
        _validate_profiles_match(active_profiles, reactive_profiles)
        raise AssertionError("ProfilesNotMatchingError was not raised for mismatched timestamps.")
    except ProfilesNotMatchingError:
        pass
