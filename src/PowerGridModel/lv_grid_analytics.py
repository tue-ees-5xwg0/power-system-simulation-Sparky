from GraphTools.graph_processing import (
    GraphCycleError,
    GraphNotFullyConnectedError,
    GraphProcessor,
    IDNotFoundError,
    IDNotUniqueError,
    InputLengthDoesNotMatchError,
)
from power_system_simulation.validate import (
    ProfilesNotMatchingError,
    ValidationException,
    _validate_active_reactive_profiles,
    _validate_ev_profile,
    _validate_feeder_connections,
    _validate_feeder_line_ids,
    _validate_load_profile,
    _validate_power_grid_model,
    _validate_source,
    _validate_transformer,
)


# Define custom exceptions for validation errors in Assignment 3
class Assignment3ValidationError(Exception):
    pass


class InvalidFeederError(Assignment3ValidationError):
    pass


class InvalidLineOutageError(Assignment3ValidationError):
    pass


class ProfileMismatchError(Assignment3ValidationError):
    pass


class TapOptimizationError(Assignment3ValidationError):
    pass


class LVGridAnalytics:
    def __init__(
            self,
            grid_path: str,
            feeder_line_ids: list[int],
            active_load_profile_path: str,
            reactive_load_profile_path: str,
            ev_profile_path: str,
    ) -> None:
        """
        Initialize the LVGridAnalytics object with the given parameters.
        Load the grid data and profiles from the specified paths.
        Validate the input data and raise exceptions if any validation fails.
        """
        self._feeder_line_ids = feeder_line_ids

        try:
            self._dataset = _validate_power_grid_model(grid_path)
            self._active_load_profiles, self._reactive_load_profiles = _validate_active_reactive_profiles(
            active_load_profile_path, reactive_load_profile_path
            )
            self._ev_pool = _validate_load_profile("EV pool", ev_profile_path)
        except ValidationException as e:
            raise Assignment3ValidationError(str(e)) from e
        except ProfilesNotMatchingError as e:
            raise ProfileMismatchError(str(e)) from e

    def validate_inputs(self) -> None:
        """Runs all the validation checks for Assignemnt 3 """

        #Checks Time indences and column IDs match between active load profile, reactive load profile, and ev profile
        _validate_ev_profile(self._active_load_profiles, self._ev_pool)
        #Check if profile IDs match the sym_load IDs in the grid
        self._validate_profile_sym_loads()
        #Checks the amount of transformers in the systems
        _validate_transformer(self._dataset)
        #Check the amount of sources in the system
        _validate_source(self._dataset)
        #Check the feeder line ids are valid
        _validate_feeder_line_ids(self._dataset, self._feeder_line_ids)
        #Checks if the feeder line ids are connected to the transformer
        _validate_feeder_connections(self._dataset, self._feeder_line_ids)
        #Check if grid is conneted and acyclic
        self._validate_topology()


    def _validate_topology(self) -> None:
        """Extracts grid data and uses GraphProcessor to validate topology."""

        # 1. Extract vertices (nodes)
        vertex_ids = self._dataset["node"]["id"].tolist()

        # 2. Extract edges (lines)
        line_data = self._dataset["line"]
        edge_ids = line_data["id"].tolist()

        # 3. Create the (from, to) pairs
        edge_vertex_id_pairs = list(zip(line_data["from_node"], line_data["to_node"], strict=True))

        # 4. Determine if edges are enabled
        # A line is only active if BOTH switches (from_status and to_status) are closed (1)
        edge_enabled = [
            bool(f_stat == 1 and t_stat == 1)
            for f_stat, t_stat in zip(line_data["from_status"], line_data["to_status"], strict=True)
        ]

        # 5. Extract the source vertex ID
        # (Assuming _validate_source already proved there is exactly 1 source)
        source_vertex_id = self._dataset["source"]["node"][0]

        # 6. Initialize GraphProcessor to trigger the automatic validation checks
        try:
            self._graph_processor = GraphProcessor(
                vertex_ids=vertex_ids,
                edge_ids=edge_ids,
                edge_vertex_id_pairs=edge_vertex_id_pairs,
                edge_enabled=edge_enabled,
                source_vertex_id=source_vertex_id
            )
        except GraphNotFullyConnectedError as e:
            raise Assignment3ValidationError("Validation failed: The base grid is not fully connected.") from e
        except GraphCycleError as e:
            raise Assignment3ValidationError("Validation failed: The base grid contains cycles.") from e
        except (IDNotFoundError, IDNotUniqueError, InputLengthDoesNotMatchError) as e:
            raise Assignment3ValidationError(f"Validation failed: Invalid graph structure - {e}") from e

    def _validate_profile_sym_loads(self) -> None:
        """Checks if the IDs in the load profiles match the actual sym_load IDs in the grid."""

        # 1. Check if the grid even has sym_loads
        if "sym_load" not in self._dataset or len(self._dataset["sym_load"]) == 0:
            raise Assignment3ValidationError("Validation failed: No sym_load components found in the grid.")

        # 2. Extract the physical sym_load IDs from the grid into a mathematical set
        grid_sym_load_ids = set(self._dataset["sym_load"]["id"].tolist())

        # 3. Extract the IDs from the profile columns
        # (Assuming the columns are purely IDs and the Timestamp is the index)
        profile_ids = set(self._active_load_profiles.columns)

        # 4. Check if they match exactly
        if profile_ids != grid_sym_load_ids:
            missing_in_grid = profile_ids - grid_sym_load_ids
            missing_in_profile = grid_sym_load_ids - profile_ids

            error_msg = "Load profile IDs do not perfectly match the grid's sym_load IDs."
            if missing_in_grid:
                error_msg += f" IDs in profile but not in grid: {missing_in_grid}."
            if missing_in_profile:
                error_msg += f" IDs in grid but not in profile: {missing_in_profile}."

            raise ProfileMismatchError(error_msg)
