from copy import deepcopy

import numpy as np
import pandas as pd
from power_grid_model import ComponentType, PowerGridModel

from GraphTools.graph_processing import EdgeAlreadyDisabledError, GraphProcessor, IDNotFoundError


class InvalidLineOutageError(Exception):
    pass


class NMinusOne:
    """
    N-1 contingency analysis for power grids.
    Analyzes what happens when a single line is disconnected and identifies
    alternative topologies that restore grid connectivity.
    """

    def __init__(
        self,
        power_grid_model_dataset: dict,
        active_load_profiles: pd.DataFrame,
        reactive_load_profiles: pd.DataFrame,
        graph_processor: GraphProcessor,
    ):
        """
        Initialize N-1 analysis.

        Args:
            power_grid_model_dataset: PGM dataset dictionary
            active_load_profiles: DataFrame with active power profiles indexed by timestamp
            reactive_load_profiles: DataFrame with reactive power profiles indexed by timestamp
            graph_processor: GraphProcessor instance for topology analysis
        """
        self._power_grid_model_dataset = power_grid_model_dataset
        self._active_load_profiles = active_load_profiles
        self._reactive_load_profiles = reactive_load_profiles
        self._graph_processor = graph_processor

    def n_minus_one(self, outage_line_id: int) -> pd.DataFrame:
        """
        Perform N-1 contingency analysis for a given line outage.

        Args:
            outage_line_id: ID of the line to be disconnected

        Returns:
            DataFrame with columns:
                - Alternative_Line_ID: Line that can be reconnected
                - Max_Loading: Maximum loading (p.u.) across all scenarios
                - Max_Loading_Line_ID: Line experiencing the maximum loading
                - Max_Loading_Timestamp: When the maximum loading occurs

        Raises:
            InvalidLineOutageError: If line doesn't exist or is already disabled
        """
        # 1. Validate the line exists and is currently active
        line_data = self._power_grid_model_dataset["line"]
        line_id_mask = np.isin(line_data["id"], outage_line_id)

        if not np.any(line_id_mask):
            raise InvalidLineOutageError(f"Line ID {outage_line_id} not found in the dataset.")

        line_idx = np.where(line_id_mask)[0][0]
        if line_data["from_status"][line_idx] != 1 or line_data["to_status"][line_idx] != 1:
            raise InvalidLineOutageError(f"Line ID {outage_line_id} is already out of service.")

        # 2. Find alternative lines that can restore connectivity
        alternatives = self._graph_processor.find_alternative_edges(outage_line_id)

        rows = []

        # 3. For each alternative, run time-series power flow
        for alt_line_id in alternatives:
            # Create a fresh copy of the dataset for this scenario
            candidate_dataset = deepcopy(self._power_grid_model_dataset)

            # Disconnect the outaged line (set both switches to 0)
            line_data = candidate_dataset["line"]
            outage_idx = np.where(np.isin(line_data["id"], outage_line_id))[0][0]
            line_data["from_status"][outage_idx] = 0
            line_data["to_status"][outage_idx] = 0

            # Connect the alternative line (set to_status to 1)
            alt_idx = np.where(np.isin(line_data["id"], alt_line_id))[0][0]
            line_data["to_status"][alt_idx] = 1

            # 4. Create batch update dataset for time-series power flow
            timestamps = self._active_load_profiles.index
            load_ids = [col for col in self._active_load_profiles.columns]

            num_timestamps = len(timestamps)
            num_loads = len(load_ids)

            # Initialize the update array with correct shape and NaN values
            from power_grid_model import initialize_array
            sym_load_updates = initialize_array("update", "sym_load", (num_timestamps, num_loads))

            # Fill in the load data
            sym_load_updates["id"] = [int(load_id) for load_id in load_ids]
            sym_load_updates["p_specified"] = self._active_load_profiles[load_ids].to_numpy()
            sym_load_updates["q_specified"] = self._reactive_load_profiles[load_ids].to_numpy()

            batch_dataset = {"sym_load": sym_load_updates}

            # 5. Run time-series power flow
            try:
                model = PowerGridModel(candidate_dataset)
                results = model.calculate_power_flow(
                    update_data=batch_dataset,
                    symmetric=True,
                )
            except Exception as e:
                # If power flow fails for this alternative, skip it
                continue

            # 6. Extract line results
            line_results = results.get(ComponentType.line, results.get("line"))
            if line_results is None:
                continue

            # Handle single vs multiple timestamps
            if line_results.ndim == 1:
                line_results = line_results[np.newaxis, :]

            if line_results.shape[0] != num_timestamps:
                continue

            # 7. Find maximum loading across all lines and timestamps
            loading = line_results["loading"]
            max_loading_overall = np.nanmax(loading)

            # Find which line and timestamp have the maximum loading
            max_idx = np.unravel_index(np.nanargmax(loading), loading.shape)
            max_loading_timestamp_idx = max_idx[0]
            max_loading_line_idx = max_idx[1]

            max_loading_line_id = int(line_results[0]["id"][max_loading_line_idx])
            max_loading_timestamp = pd.to_datetime(timestamps[max_loading_timestamp_idx])

            # 8. Store results
            rows.append({
                "Alternative_Line_ID": alt_line_id,
                "Max_Loading": max_loading_overall,
                "Max_Loading_Line_ID": max_loading_line_id,
                "Max_Loading_Timestamp": max_loading_timestamp,
            })

        # Return results DataFrame
        return pd.DataFrame(rows, columns=[
            "Alternative_Line_ID",
            "Max_Loading",
            "Max_Loading_Line_ID",
            "Max_Loading_Timestamp",
        ])
