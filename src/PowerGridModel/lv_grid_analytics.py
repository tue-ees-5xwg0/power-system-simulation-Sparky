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
from PowerGridModel.tap_position_optimization import TapOptimizationError, TapPositionOptimization

__all__ = [
    "Assignment3ValidationError",
    "InvalidFeederError",
    "InvalidLineOutageError",
    "LVGridAnalytics",
    "ProfileMismatchError",
    "TapOptimizationError",
]


# Define custom exceptions for validation errors in Assignment 3
class Assignment3ValidationError(Exception):
    pass


class InvalidFeederError(Assignment3ValidationError):
    pass


class InvalidLineOutageError(Assignment3ValidationError):
    pass


class ProfileMismatchError(Assignment3ValidationError):
    pass


class LVGridAnalytics(TapPositionOptimization):
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
            self._ev_pool = validate_load_profile("EV pool", ev_profile_path)
        except ValidationException as e:
            raise Assignment3ValidationError(str(e)) from e
        except ProfilesNotMatchingError as e:
            raise ProfileMismatchError(str(e)) from e

        TapPositionOptimization.__init__(
            self,
            dataset=self._dataset,
            active_load_profiles=self._active_load_profiles,
            reactive_load_profiles=self._reactive_load_profiles,
        )

    def validate_inputs(self) -> None:
        """Runs all the validation checks for Assignment 3."""

        self._validate_ev_profile()
        # TODO: Add the remaining Assignment 3 validation checks.

    def _validate_ev_profile(self) -> None:
        if not self._active_load_profiles.index.equals(self._ev_pool.index):
            raise ProfileMismatchError("Active load profile and EV profile have different time indices.")

        if len(self._ev_pool.columns) < len(self._active_load_profiles.columns):
            raise ProfileMismatchError("EV profile has fewer columns than the active load profile.")
