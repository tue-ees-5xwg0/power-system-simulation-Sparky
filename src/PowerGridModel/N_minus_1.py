
import numpy as np
import pandas as pd
from pandas import DataFrame
from power_grid_model import ComponentType, PowerGridModel
from GraphTools.graph_processing import find_alternative_edges


class InvalidLineOutageError(Exception):
    pass

def _create_power_grid_model(self) -> PowerGridModel:
    pass

class NMinusOne:
    def __init__(self, power_grid_model_dataset):
        self._power_grid_model_dataset = power_grid_model_dataset
        self._dataset = self._create_power_grid_model()
        self._active_load_profiles = self._dataset.load_profiles

    def _create_power_grid_model(self) -> PowerGridModel:
        pass

    def n_minus_one(self, outage_line_id):
        self._validate_inputs()
        # 1. validate line exists
        # 2. check from_status & to_status == 1

        # 3. create base dataset copy
        base_dataset = copy(...)

        # 4. disconnect outage line
        set from_status/to_status = 0

        # 5. find alternative edges
        alternatives = find_alternative_edges(...)

        rows = []

        for alt_id in alternatives:

            # 6. copy dataset again
            candidate_dataset = copy(base_dataset)

            # 7. connect alternative line
            set to_status = 1

            # 8. run time-series PF
            results = run_pf(candidate_dataset)

            # 9. compute:
            #    max loading
            #    line ID
            #    timestamp

            rows.append({...})
    return pd.DataFrame(rows, columns=[...])
