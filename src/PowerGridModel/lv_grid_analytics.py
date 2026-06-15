from PowerGridModel.power_grid_calculator import (
    ProfilesNotMatchingError,
    ValidationException,
    _validate_active_reactive_profiles,
    _validate_load_profile,
    _validate_power_grid_model,
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

            self._validate_ev_profile()
            #TODO Add more validation checks

        def _validate_ev_profile(self) -> None:
            #Check if the ev profile match the loads
            if not self._active_profiles.index.equals(self._ev_pool.index):
                raise ProfilesNotMatchingError("Active load profile and EV profile have different time indices.")

            #Check if there are enough columns in the ev profile
            if len(self._ev_pool.columns) < len(self._active_profiles.columns):
                raise ProfilesNotMatchingError("EV profile has fewer columns than the active load profile.")

        def validate_transformer(self, transformer_id: int) -> None:
            #Check if the transformer id is valid
            if transformer_id not in self._dataset.transformers.index:
                raise InvalidFeederError(f"Transformer ID {transformer_id} is not valid.")
            # check if there is only 1 transformer in the system
            if len(self._dataset.transformers) != 1:
                raise InvalidFeederError("There should only be one transformer in the system. Please specify the transformer ID.")

        # There should be exactly one source in the system
        def validate_source(self) -> None:
            if len(self._dataset.sources) != 1:
                raise InvalidFeederError("There should be exactly one source in the system.")

            # check if Every LV feeder ID is a valid line ID.
        def validate_feeder_line_ids(self) -> None:
            for line_id in self._feeder_line_ids:
                if line_id not in self._dataset.lines.index:
                    raise InvalidFeederError(f"Feeder line ID {line_id} is not valid.")

            # check if Every feeder line has from_node == transformer.to_node
        def validate_feeder_connections(self) -> None:
            transformer_to_node = self._dataset.transformers.iloc[0].to_node
            for line_id in self._feeder_line_ids:
                line = self._dataset.lines.loc[line_id]
                if line.from_node != transformer_to_node:
                    raise InvalidFeederError(f"Feeder line ID {line_id} is not connected to the transformer.")
