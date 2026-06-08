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
