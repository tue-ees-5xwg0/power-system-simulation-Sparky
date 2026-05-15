from PowerGridModel.power_grid_calculator import ValidationException, _InputValidator

FILE_PATH_VALID_INPUT = "tests/PGM_TestData/input"
FILE_PATH_FALSE_INPUT = "tests/PGM_TestData/false_input"

def test_validate_power_grid_model():
    # Test with valid input
    try:
        _InputValidator._validate_power_grid_model(FILE_PATH_VALID_INPUT + "/input_network_data.json")
    except ValidationException:
        raise AssertionError("ValidationException was raised for valid input.") from None

    # Test with invalid file path
    try:
        _InputValidator._validate_power_grid_model(FILE_PATH_FALSE_INPUT + "/invalid_path.json")
        raise AssertionError("ValidationException was not raised for invalid file path.")
    except ValidationException:
        pass

    # Test with invalid file extension
    try:
        _InputValidator._validate_power_grid_model(FILE_PATH_FALSE_INPUT + "/wrong.extension")
        raise AssertionError("ValidationException was not raised for invalid file extension.")
    except ValidationException:
        pass

    # Test with inconsistent data
    try:
        _InputValidator._validate_power_grid_model(FILE_PATH_FALSE_INPUT + "/inconsistent_data.json")
        raise AssertionError("ValidationException was not raised for inconsistent data.")
    except ValidationException:
        pass

def test_validate_load_profiles():
    # Test with valid input
    try:
        _InputValidator._validate_load_profiles(
            FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
            FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet"
        )
    except ValidationException:
        raise AssertionError("ValidationException was raised for valid input.") from None

    # Test with invalid file path
    try:
        _InputValidator._validate_load_profiles(
            FILE_PATH_FALSE_INPUT + "/invalid_active_path.parquet",
            FILE_PATH_FALSE_INPUT + "/invalid_reactive_path.parquet"
        )
        raise AssertionError("ValidationException was not raised for invalid file path.")
    except ValidationException:
        pass

    # Test with invalid file extension
    try:
        _InputValidator._validate_load_profiles(
            FILE_PATH_FALSE_INPUT + "/wrong.extension",
            FILE_PATH_FALSE_INPUT + "/wrong.extension"
        )
        raise AssertionError("ValidationException was not raised for invalid file extension.")
    except ValidationException:
        pass
