# power-system-simulation

This is a student project for Power System Simulation.

It provides a comprehensive set of tools for Low Voltage (LV) grid analytics, including EV penetration studies, N-1 contingency analysis, and optimal tap position calculation, built on top of `power-grid-model`.

## Library Components

### Core Classes

* **`LVGridAnalytics`** (`lv_grid_analytics.py`): The primary orchestrator class for analyzing LV grids. It loads and validates the grid dataset, profiles, and EV data. It exposes methods to run N-1 analysis (`n_minus_one`), EV penetration (`run_ev_penetration`), and inherits functionality for tap position optimization.
* **`GraphProcessor`** (`graph_processing.py`): A graph utility class that validates the grid's topology (ensuring it is fully connected and acyclic in its base state). It provides methods to find downstream vertices (for EV assignments) and alternative edges (for N-1 contingency restoration).
* **`NMinusOne`** (`N_minus_1.py`): Responsible for executing N-1 contingency simulations. Given a disabled line, it identifies alternative valid lines to reconnect the grid and computes the maximum loading across all viable alternative topologies using time-series power flow.
* **`TapPositionOptimization`** (`tap_position_optimization.py`): Base class of `LVGridAnalytics` handling optimal transformer tap position evaluations. Iterates through possible tap positions and returns the best state based on a custom or provided criterion.
* **`GridModel`** (`power_grid_calculator.py`): Wraps the `PowerGridModel` initialization and batch update execution, providing a streamlined way to extract time-series results for node voltages and line loading.

### Helper Functions

* **Tap Optimization Criteria** (`tap_position_optimization.py`):
  * `minimize_total_loss`: Evaluates tap results to minimize the total system kWh loss over the analyzed period.
  * `minimize_average_voltage_deviation`: Evaluates tap results to minimize the average voltage deviation from 1.0 p.u.
* **EV Penetration** (`ev_penetration.py`):
  * `run_ev_penetration`: Randomly assigns EV charging profiles to a specific percentage of households downstream of targeted feeders, calculates the time-series power flow, and aggregates the impact on lines and nodes.
* **Validation Utilities** (`validate.py`):
  * `validate_power_grid_model`: Loads, parses, and validates JSON datasets into the standard `power-grid-model` format.
  * `validate_active_reactive_profiles` & `validate_load_profile`: Securely load and validate `.parquet` load profiles.
  * `validate_profiles_match`: Ensures dimensions, column indices, and timestamps match exactly between active and reactive datasets.

### Exceptions

The library defines a granular exception hierarchy to capture domain-specific errors:

* **Validation & Data Integrity**:
  * `ValidationException`: Raised when file paths are invalid, missing, or malformed.
  * `ProfilesNotMatchingError` / `ProfileMismatchError`: Raised when timestamp indices, load IDs, or column counts misalign between load profiles and grid datasets.
  * `Assignment3ValidationError`: A top-level wrapper exception for structural validation errors found before running analytics (e.g., incorrect transformer/source counts, disjoint networks).
* **Graph Processing Errors** (`graph_processing.py`):
  * `GraphNotFullyConnectedError` / `GraphCycleError`: Enforces that the grid's enabled lines form a valid tree topology initially.
  * `IDNotFoundError` / `IDNotUniqueError` / `InputLengthDoesNotMatchError`: Raised for improper graph initialization arrays.
  * `EdgeAlreadyDisabledError`: Triggered when attempting to disable an edge that is already out of service.
* **Simulation-Specific Errors**:
  * `InvalidLineOutageError`: Raised during N-1 analysis when targeting a line ID that does not exist or is already disabled.
  * `TapOptimizationError`: Raised when tap settings are missing on the transformer, misconfigured (e.g., `tap_min` > `tap_max`), or when power flow fails during an optimization loop.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

In the root of the repository, sync all dependencies using:

```shell
uv sync
```

After installation, run the test.

```shell
uv run pytest
```

## Code style and quality check

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

You can run the following command to check and auto-fix code issues:

```shell
uv run ruff check --fix .
```

You can run the following command to format your code:

```shell
uv run ruff format .
```

## Working with Jupyter Notebooks

Jupyter notebooks in the `example/` folder can be opened directly in VS Code. The project includes `ipykernel` in the development dependencies, which allows VS Code to run notebook cells using the `.venv` environment.

## Folder structure of the repository

The folder structure of the repository is explained as below.

* [`src/power_system_simulation`](./src/power_system_simulation) is the main folder of the package. You should put your new functionality code there.
* [`tests`](./tests) is the folder containing the test files. You should put your test code there.
* [`example`](./example) contains the example notebook. You should modify the notebook for your presentation.
* [`.vscode`](./.vscode) contains the setting file for the IDE VSCode.
* [`.github/workflows`](./.github/workflows) contains the continuous integration (CI) configurations.
