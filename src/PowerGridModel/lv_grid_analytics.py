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
    validate_active_reactive_profiles,
    validate_load_profile,
    validate_power_grid_model,
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
            self._dataset = validate_power_grid_model(grid_path)
            self._active_load_profiles, self._reactive_load_profiles = validate_active_reactive_profiles(
            active_load_profile_path, reactive_load_profile_path
            )
            self._ev_pool = validate_load_profile("EV pool", ev_profile_path)
        except ValidationException as e:
            raise Assignment3ValidationError(str(e)) from e
        except ProfilesNotMatchingError as e:
            raise ProfileMismatchError(str(e)) from e

    def validate_inputs(self) -> None:
        """Runs all the validation checks for Assignemnt 3 """

        #Checks Time indences and column IDs match between active load profile, reactive load profile, and ev profile
        self._validate_ev_profile()
        #Check if profile IDs match the sym_load IDs in the grid
        self._validate_profile_sym_loads()
        #Checks the amount of transformers in the systems
        self._validate_transformer()
        #Check the amount of sources in the system
        self._validate_source()
        #Check the feeder line ids are valid
        self._validate_feeder_line_ids()
        #Checks if the feeder line ids are connected to the transformer
        self._validate_feeder_connections()
        #Check if grid is conneted and acyclic
        self._validate_topology()


    def _validate_topology(self) -> None:
        """Extracts grid data and uses GraphProcessor to validate topology."""

        # 1. Extract vertices (nodes)
        vertex_ids = [int(v) for v in self._dataset["node"]["id"]]

        # 2. Extract edges (lines and transformers)
        line_data = self._dataset["line"]
        transformer_data = self._dataset["transformer"]
        edge_ids = [int(e) for e in line_data["id"]] + [int(e) for e in transformer_data["id"]]

        # 3. Create the (from, to) pairs
        edge_vertex_id_pairs = [
            (int(f), int(t)) for f, t in zip(line_data["from_node"], line_data["to_node"], strict=True)
        ] + [
            (int(f), int(t)) for f, t in zip(transformer_data["from_node"], transformer_data["to_node"], strict=True)
        ]

        # 4. Determine if edges are enabled
        # A line is only active if BOTH switches (from_status and to_status) are closed (1)
        edge_enabled = [
            bool(f_stat == 1 and t_stat == 1)
            for f_stat, t_stat in zip(line_data["from_status"], line_data["to_status"], strict=True)
        ] + [True] * len(transformer_data)  # Transformers are always enabled

        # 5. Extract the source vertex ID
        # (Assuming _validate_source already proved there is exactly 1 source)
        source_vertex_id = int(self._dataset["source"]["node"][0])

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

    def _validate_ev_profile(self) -> None:
        #Check if the ev profile match the loads
        if not self._active_load_profiles.index.equals(self._ev_pool.index):
            raise ProfilesNotMatchingError("Active load profile and EV profile have different time indices.")

        #Check if there are enough columns in the ev profile
        if len(self._ev_pool.columns) < len(self._active_load_profiles.columns):
            raise ProfilesNotMatchingError("EV profile has fewer columns than the active load profile.")

    def _validate_transformer(self) -> None:
        # check if there is only 1 transformer in the system
        if len(self._dataset["transformer"]) != 1:
            raise ValidationException(
                "There should only be one transformer in the system. Please specify the transformer ID.")

    # There should be exactly one source in the system
    def _validate_source(self) -> None:
        if len(self._dataset["source"]) != 1:
            raise ValidationException("There should be exactly one source in the system.")

    # check if Every LV feeder ID is a valid line ID.
    def _validate_feeder_line_ids(self) -> None:
        for line_id in self._feeder_line_ids:
            if line_id not in self._dataset["line"]["id"]:
                raise ValidationException(f"Feeder line ID {line_id} is not valid.")

    # check if Every feeder line has from_node == transformer.to_node
    def _validate_feeder_connections(self) -> None:
        transformer_to_node = self._dataset["transformer"]["to_node"][0]
        line_from_node_dict = dict(zip(self._dataset["line"]["id"], self._dataset["line"]["from_node"], strict=True))
        for line_id in self._feeder_line_ids:
            if line_from_node_dict.get(line_id) != transformer_to_node:
                raise ValidationException(f"Feeder line ID {line_id} is not connected to the transformer.")
