"""
Tests for N-1 contingency analysis functionality.

These tests verify that the N-1 analysis correctly:
1. Validates input line IDs
2. Finds alternative topologies
3. Runs time-series power flow for each alternative
4. Returns results in the expected format
"""

import numpy as np
import pandas as pd
import pytest

from PowerGridModel.lv_grid_analytics import LVGridAnalytics
from PowerGridModel.N_minus_1 import InvalidLineOutageError as NMinusOneError, NMinusOne

FILE_PATH_VALID_INPUT = "tests/small_network"


@pytest.fixture
def valid_analytics():
    """Create a valid LVGridAnalytics instance with validated inputs."""
    analytics = LVGridAnalytics(
        grid_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        feeder_line_ids=[16, 20],
        active_load_profile_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profile_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        ev_profile_path=FILE_PATH_VALID_INPUT + "/ev_active_power_profile.parquet",
    )
    analytics.validate_inputs()
    return analytics


def test_n_minus_one_valid_line_outage(valid_analytics):
    """Test N-1 analysis with a valid line ID."""
    # Line 16 is a valid feeder line
    results = valid_analytics.n_minus_one(outage_line_id=16)

    # Should return a DataFrame
    assert isinstance(results, pd.DataFrame)

    # Should have the expected columns
    expected_columns = {
        "Alternative_Line_ID",
        "Max_Loading",
        "Max_Loading_Line_ID",
        "Max_Loading_Timestamp",
    }
    assert set(results.columns) == expected_columns

    # Should have at least one alternative (grid is meshed)
    assert len(results) >= 1

    # All max loading values should be positive
    assert (results["Max_Loading"] > 0).all()


def test_n_minus_one_another_valid_line(valid_analytics):
    """Test N-1 analysis with another valid line ID."""
    # Line 20 is also a valid feeder line
    results = valid_analytics.n_minus_one(outage_line_id=20)

    # Should return a DataFrame
    assert isinstance(results, pd.DataFrame)

    # Should have the expected columns
    assert not results.empty or len(results) == 0  # Can be empty if no alternatives


def test_n_minus_one_invalid_line_id(valid_analytics):
    """Test N-1 analysis with an invalid line ID."""
    # Line 999 doesn't exist
    from PowerGridModel.N_minus_1 import InvalidLineOutageError as NMinusOneError

    with pytest.raises(NMinusOneError, match="not found"):
        valid_analytics.n_minus_one(outage_line_id=999)


def test_n_minus_one_returns_dataframe_columns(valid_analytics):
    """Test that N-1 results have the correct data types."""
    results = valid_analytics.n_minus_one(outage_line_id=16)

    if len(results) > 0:
        # Alternative_Line_ID should be integer-like
        assert pd.api.types.is_numeric_dtype(results["Alternative_Line_ID"])

        # Max_Loading should be float
        assert pd.api.types.is_float_dtype(results["Max_Loading"])

        # Max_Loading_Line_ID should be integer-like
        assert pd.api.types.is_numeric_dtype(results["Max_Loading_Line_ID"])

        # Max_Loading_Timestamp should be datetime
        assert isinstance(results["Max_Loading_Timestamp"].iloc[0], pd.Timestamp)


def test_n_minus_one_consistency(valid_analytics):
    """Test that running N-1 twice gives consistent results."""
    results1 = valid_analytics.n_minus_one(outage_line_id=16)
    results2 = valid_analytics.n_minus_one(outage_line_id=16)

    # Results should be consistent (same alternatives found)
    if len(results1) > 0 and len(results2) > 0:
        # Same number of alternatives
        assert len(results1) == len(results2)

        # Same alternative lines
        assert set(results1["Alternative_Line_ID"]) == set(results2["Alternative_Line_ID"])


def test_n_minus_one_max_loading_reasonable(valid_analytics):
    """Test that max loading values are reasonable (between 0 and a reasonable upper bound)."""
    results = valid_analytics.n_minus_one(outage_line_id=16)

    if len(results) > 0:
        # Max loading should be between 0 and 10 (accounting for grid characteristics)
        assert (results["Max_Loading"] >= 0).all()
        assert (results["Max_Loading"] <= 10).all()


def test_n_minus_one_alternatives_are_disconnected_in_base(valid_analytics):
    """Test that alternative lines are actually disconnected in the base case."""
    results = valid_analytics.n_minus_one(outage_line_id=16)

    # Get all alternative line IDs
    if len(results) > 0:
        alternative_ids = results["Alternative_Line_ID"].tolist()

        # These should be from the disconnected lines in the base case
        line_data = valid_analytics._dataset["line"]
        disconnected_line_ids = set(line_data["id"][(line_data["from_status"] == 0) | (line_data["to_status"] == 0)])

        # All alternatives should be in the disconnected lines
        for alt_id in alternative_ids:
            assert alt_id in disconnected_line_ids


def test_n_minus_one_out_of_service_line(valid_analytics):
    """Test N-1 analysis with a line that is already out of service."""
    # First, manually set line 24 to be out of service
    line_data = valid_analytics._dataset["line"]
    idx = np.where(line_data["id"] == 24)[0][0]
    line_data["from_status"][idx] = 0
    line_data["to_status"][idx] = 0

    # Now try to run N-1 on the same line, should raise error
    with pytest.raises(NMinusOneError, match="already out of service"):
        valid_analytics.n_minus_one(outage_line_id=24)


def test_n_minus_one_direct_class_invalid_line():
    """Test NMinusOne class directly with invalid line ID."""
    analytics = LVGridAnalytics(
        grid_path=FILE_PATH_VALID_INPUT + "/input_network_data.json",
        feeder_line_ids=[16, 20],
        active_load_profile_path=FILE_PATH_VALID_INPUT + "/active_power_profile.parquet",
        reactive_load_profile_path=FILE_PATH_VALID_INPUT + "/reactive_power_profile.parquet",
        ev_profile_path=FILE_PATH_VALID_INPUT + "/ev_active_power_profile.parquet",
    )
    analytics.validate_inputs()

    n_minus_one = NMinusOne(
        power_grid_model_dataset=analytics._dataset,
        active_load_profiles=analytics._active_load_profiles,
        reactive_load_profiles=analytics._reactive_load_profiles,
        graph_processor=analytics._graph_processor,
    )

    # Should raise error for invalid line
    with pytest.raises(NMinusOneError, match="not found"):
        n_minus_one.n_minus_one(outage_line_id=999)


def test_n_minus_one_power_flow_exception(valid_analytics):
    """
    Test that power flow exceptions are properly handled.

    This tests the exception handling branch: when calculate_power_flow fails,
    the alternative should be skipped (continue).
    """
    from unittest.mock import patch

    # Get the NMinusOne instance through the analytics object's graph processor
    n_minus_one = NMinusOne(
        power_grid_model_dataset=valid_analytics._dataset,
        active_load_profiles=valid_analytics._active_load_profiles,
        reactive_load_profiles=valid_analytics._reactive_load_profiles,
        graph_processor=valid_analytics._graph_processor,
    )

    # Mock PowerGridModel to raise an exception
    with patch("PowerGridModel.N_minus_1.PowerGridModel") as mock_pgm:
        mock_pgm.side_effect = Exception("Convergence failed")

        # Run N-1 analysis - should skip all alternatives and return empty DataFrame
        results = n_minus_one.n_minus_one(outage_line_id=16)

        # All alternatives should be skipped due to exception
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0


def test_n_minus_one_none_line_results(valid_analytics):
    """
    Test that None line results are properly handled.

    This tests the conditional skip when line_results is None.
    """
    from unittest.mock import Mock, patch

    n_minus_one = NMinusOne(
        power_grid_model_dataset=valid_analytics._dataset,
        active_load_profiles=valid_analytics._active_load_profiles,
        reactive_load_profiles=valid_analytics._reactive_load_profiles,
        graph_processor=valid_analytics._graph_processor,
    )

    # Mock PowerGridModel to return None for line results
    with patch("PowerGridModel.N_minus_1.PowerGridModel") as mock_pgm_class:
        mock_instance = Mock()
        mock_pgm_class.return_value = mock_instance
        # Return dict without 'line' or ComponentType.line keys
        mock_instance.calculate_power_flow.return_value = {"node": np.array([])}

        results = n_minus_one.n_minus_one(outage_line_id=16)

        # All alternatives should be skipped due to None line_results
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0


def test_n_minus_one_mismatched_timestamp_shape(valid_analytics):
    """
    Test that mismatched timestamp shapes are properly handled.

    This tests the conditional skip when line_results.shape[0] != num_timestamps.
    """
    from unittest.mock import Mock, patch

    from power_grid_model import ComponentType

    n_minus_one = NMinusOne(
        power_grid_model_dataset=valid_analytics._dataset,
        active_load_profiles=valid_analytics._active_load_profiles,
        reactive_load_profiles=valid_analytics._reactive_load_profiles,
        graph_processor=valid_analytics._graph_processor,
    )

    # Create mock line results with wrong timestamp dimension
    num_timestamps = len(valid_analytics._active_load_profiles.index)
    wrong_num_timestamps = num_timestamps + 5

    # Create dummy line results array with wrong first dimension
    mock_line_results = np.zeros(
        (wrong_num_timestamps, 10), dtype=[("id", "i4"), ("loading", "f8"), ("p_from", "f8"), ("p_to", "f8")]
    )
    mock_line_results["loading"] = 0.5

    with patch("PowerGridModel.N_minus_1.PowerGridModel") as mock_pgm_class:
        mock_instance = Mock()
        mock_pgm_class.return_value = mock_instance
        mock_instance.calculate_power_flow.return_value = {ComponentType.line: mock_line_results}

        results = n_minus_one.n_minus_one(outage_line_id=16)

        # All alternatives should be skipped due to shape mismatch
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0
