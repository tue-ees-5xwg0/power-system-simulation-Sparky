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
        #Checks the amount of transformers in the systems
        _validate_transformer(self._dataset)
        #Check the amount of sources in the system
        _validate_source(self._dataset)
        #Check the feeder line ids are valid
        _validate_feeder_line_ids(self._dataset, self._feeder_line_ids)
        #Checks if the feeder line ids are connected to the transformer
        _validate_feeder_connections(self._dataset, self._feeder_line_ids)
        #TODO Add more validation checks
