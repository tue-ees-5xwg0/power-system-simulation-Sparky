# Assignment 3 Class and API Structure

## Summary

Assignment 3 should add low-voltage grid analytics on top of the existing Assignment 1 graph tools and Assignment 2 power-flow calculator.

The implementation should introduce one high-level API class, `LVGridAnalytics`, that owns input loading, validation, graph conversion, and the three required analysis workflows:

- EV penetration simulation
- Transformer tap-position optimization
- N-1 topology analysis

The Assignment 3 functionality should live in the existing `PowerGridModel` package, because `pyproject.toml` currently packages `src/PowerGridModel`.

Recommended new module:

```text
src/PowerGridModel/lv_grid_analytics.py
```

The existing Assignment 2 `GridModel` API should remain compatible with current tests.

## Public API

### Main class

```python
class LVGridAnalytics:
    def __init__(
        self,
        grid_path: str,
        feeder_line_ids: list[int],
        active_load_profile_path: str,
        reactive_load_profile_path: str,
        ev_profile_pool_path: str,
    ) -> None:
        ...

    def validate_inputs(self) -> None:
        ...

    def run_ev_penetration(
        self,
        penetration_level: float,
        random_seed: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def optimize_tap_position(
        self,
        criterion: TapOptimizationCriterion,
    ) -> TapOptimizationResult:
        ...

    def run_n_minus_one(
        self,
        outage_line_id: int,
    ) -> pd.DataFrame:
        ...
```

### Tap optimization types

```python
from dataclasses import dataclass
from enum import StrEnum


class TapOptimizationCriterion(StrEnum):
    MIN_LOSS = "min_loss"
    MIN_VOLTAGE_DEVIATION = "min_voltage_deviation"


@dataclass(frozen=True)
class TapOptimizationResult:
    tap_position: int
    criterion: TapOptimizationCriterion
    total_loss_kwh: float
    average_voltage_deviation_pu: float
    all_tap_results: pd.DataFrame
```

`all_tap_results` should contain one row per tested tap position with these columns:

```text
tap_position
total_loss_kwh
average_voltage_deviation_pu
```

### Exceptions

Add Assignment 3-specific exceptions so calling code can distinguish assignment-level validation errors from low-level PGM errors.

```python
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
```

PGM validation errors may be passed through directly or wrapped in `Assignment3ValidationError`, but the behavior should be consistent across the module.

## Data Model and Internal Helpers

### Input data

`LVGridAnalytics.__init__` should load and store:

- PGM grid JSON as a PGM dataset
- LV feeder line IDs as `list[int]`
- active load profiles as `pd.DataFrame`
- reactive load profiles as `pd.DataFrame`
- EV profile pool as `pd.DataFrame`

Active and reactive load profile columns are `sym_load` IDs.

EV profile pool columns are anonymous EV profile IDs or sequence numbers. They are not expected to match `sym_load` IDs.

### Grid inventory

Create an internal structure to make validation and graph conversion easier.

```python
@dataclass(frozen=True)
class GridInventory:
    node_ids: list[int]
    line_ids: list[int]
    line_from_to: dict[int, tuple[int, int]]
    line_enabled: dict[int, bool]
    source_id: int
    source_node_id: int
    transformer_id: int
    transformer_from_node: int
    transformer_to_node: int
    sym_load_ids: list[int]
    sym_load_node_by_id: dict[int, int]
```

`line_enabled` should be `True` only when both `from_status` and `to_status` are `1`.

### Helper functions

Recommended private helpers:

```python
def _load_grid_dataset(grid_path: str) -> Dataset:
    ...

def _load_profile(path: str, label: str) -> pd.DataFrame:
    ...

def _extract_grid_inventory(dataset: Dataset) -> GridInventory:
    ...

def _validate_profile_alignment(
    active: pd.DataFrame,
    reactive: pd.DataFrame,
    ev_pool: pd.DataFrame,
    sym_load_ids: list[int],
) -> None:
    ...

def _build_graph_processor(
    inventory: GridInventory,
) -> GraphProcessor:
    ...

def _create_batch_update(
    active_profile: pd.DataFrame,
    reactive_profile: pd.DataFrame,
) -> dict:
    ...

def _run_time_series_power_flow(
    dataset: Dataset,
    active_profile: pd.DataFrame,
    reactive_profile: pd.DataFrame,
) -> dict:
    ...

def _aggregate_results(
    results: dict,
    timestamps: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...

def _copy_dataset_with_line_status(
    dataset: Dataset,
    line_id: int,
    from_status: int,
    to_status: int,
) -> Dataset:
    ...

def _copy_dataset_with_tap_position(
    dataset: Dataset,
    transformer_id: int,
    tap_position: int,
) -> Dataset:
    ...
```

The time-series power-flow and aggregation helpers can reuse or refactor existing Assignment 2 code from `power_grid_calculator.py`.

## Input Validation

`validate_inputs()` must check all Assignment 3 input criteria.

Required checks:

- LV grid JSON exists, is readable, deserializes as PGM input data, and can construct a `PowerGridModel`.
- The grid has exactly one `transformer`.
- The grid has exactly one `source`.
- Every LV feeder ID is a valid line ID.
- Every feeder line has `from_node == transformer.to_node`.
- The enabled base grid is fully connected.
- The enabled base grid has no cycles.
- Active and reactive load profile timestamps match.
- Active and reactive load profile IDs match.
- Active and reactive load profile IDs are valid `sym_load` IDs.
- EV profile pool timestamps match the load profile timestamps.
- The number of EV profile columns is at least the number of `sym_load` IDs.

Graph validation should use Assignment 1:

```python
GraphProcessor(
    vertex_ids=inventory.node_ids,
    edge_ids=inventory.line_ids,
    edge_vertex_id_pairs=[
        inventory.line_from_to[line_id]
        for line_id in inventory.line_ids
    ],
    edge_enabled=[
        inventory.line_enabled[line_id]
        for line_id in inventory.line_ids
    ],
    source_vertex_id=inventory.source_node_id,
)
```

If `GraphProcessor` raises `GraphNotFullyConnectedError` or `GraphCycleError`, wrap or pass through the error as an Assignment 3 validation failure.

## EV Penetration Workflow

Public method:

```python
def run_ev_penetration(
    self,
    penetration_level: float,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Behavior:

1. Call `validate_inputs()`.
2. Validate `0.0 <= penetration_level <= 1.0`.
3. Compute the number of EVs per feeder:

```python
evs_per_feeder = math.floor(
    penetration_level * total_number_of_houses / number_of_feeders
)
```

4. Build a `GraphProcessor` from the base grid.
5. For each feeder line ID:
   - call `find_downstream_vertices(feeder_line_id)`
   - map downstream node IDs to `sym_load` IDs
   - randomly select `evs_per_feeder` houses from those `sym_load` IDs
6. Randomly assign one EV profile to each selected house without replacement.
7. Add each assigned EV active-power curve to the matching `sym_load` active load profile.
8. Do not change reactive load profiles, because EV reactive power is assumed to be zero.
9. Run time-series power flow.
10. Return the two Assignment 2 aggregation tables:
    - row per timestamp
    - row per line

Randomness should use:

```python
rng = np.random.default_rng(random_seed)
```

This keeps tests reproducible without changing global NumPy random state.

If a feeder has fewer houses than `evs_per_feeder`, raise `Assignment3ValidationError` with a clear message.

## Tap Optimization Workflow

Public method:

```python
def optimize_tap_position(
    self,
    criterion: TapOptimizationCriterion,
) -> TapOptimizationResult:
    ...
```

Behavior:

1. Call `validate_inputs()`.
2. Use the original household active and reactive load profiles only.
3. Do not include EV profiles in this calculation.
4. Find the single transformer ID from `GridInventory`.
5. Discover all possible tap positions from the transformer data.
6. For every possible tap position:
   - copy the original dataset
   - set the transformer tap position
   - run time-series power flow
   - aggregate results
   - compute `total_loss_kwh`
   - compute `average_voltage_deviation_pu`
7. Select the best tap position according to `criterion`.
8. Return `TapOptimizationResult`.

Loss metric:

```python
total_loss_kwh = line_table["Total_Loss"].sum()
```

Voltage deviation metric:

```python
average_voltage_deviation_pu = (
    (timestamp_table["Max_Voltage"] - 1.0).abs()
    + (timestamp_table["Min_Voltage"] - 1.0).abs()
).mean() / 2
```

If the transformer does not expose enough tap-position information to enumerate tap positions, raise `TapOptimizationError`.

## N-1 Analysis Workflow

Public method:

```python
def run_n_minus_one(
    self,
    outage_line_id: int,
) -> pd.DataFrame:
    ...
```

Behavior:

1. Call `validate_inputs()`.
2. Check that `outage_line_id` is a valid line ID.
3. Check that the line is connected on both sides in the base case:

```python
from_status == 1
to_status == 1
```

4. If the line does not exist or is already disconnected, raise `InvalidLineOutageError`.
5. Build a `GraphProcessor` from the base grid.
6. Call:

```python
alternative_line_ids = graph_processor.find_alternative_edges(outage_line_id)
```

7. For each alternative line:
   - copy the original dataset
   - disconnect the outage line by setting both `from_status` and `to_status` to `0`
   - reconnect the alternative line by setting `to_status` to `1`
   - run time-series power flow
   - find the maximum loading across all lines and timestamps
8. Return one summary row per alternative.

Output table:

```text
Index name: alternative_line_id

Columns:
- max_loading
- max_loading_line_id
- max_loading_timestamp
```

If there are no alternatives, return an empty `DataFrame` with the same columns and index name.

## Expected Output Shapes

### EV penetration output

Return the same two tables as Assignment 2:

```python
timestamp_table, line_table = analytics.run_ev_penetration(
    penetration_level=0.2,
    random_seed=0,
)
```

### Tap optimization output

```python
result = analytics.optimize_tap_position(
    criterion=TapOptimizationCriterion.MIN_LOSS,
)

result.tap_position
result.total_loss_kwh
result.average_voltage_deviation_pu
result.all_tap_results
```

### N-1 output

```python
n_minus_one_table = analytics.run_n_minus_one(outage_line_id=123)
```

Example shape:

```text
alternative_line_id | max_loading | max_loading_line_id | max_loading_timestamp
--------------------|-------------|---------------------|----------------------
456                 | 0.82        | 111                 | 2026-01-01 18:15:00
789                 | 0.91        | 222                 | 2026-01-01 18:30:00
```

## Testing Plan

Use the provided `small_network` dataset for tests.

### Validation tests

Test that:

- valid input passes `validate_inputs()`
- invalid feeder ID raises `InvalidFeederError`
- feeder line not connected to transformer LV node raises `InvalidFeederError`
- timestamp mismatch between active and reactive profiles raises `ProfileMismatchError`
- timestamp mismatch between load profiles and EV pool raises `ProfileMismatchError`
- active/reactive load ID mismatch raises `ProfileMismatchError`
- load profile ID not present in `sym_load` raises `ProfileMismatchError`
- too few EV profiles raises `ProfileMismatchError`
- disconnected base grid raises validation error
- cyclic base grid raises validation error

### EV penetration tests

Test that:

- `random_seed` makes EV assignment reproducible
- EV profiles are not assigned more than once
- `penetration_level=0.0` leaves active profiles unchanged
- reactive load profiles are unchanged after EV assignment
- returned tables have the same schema as Assignment 2 aggregation tables

### Tap optimization tests

Test that:

- `TapOptimizationCriterion.MIN_LOSS` returns a valid `TapOptimizationResult`
- `TapOptimizationCriterion.MIN_VOLTAGE_DEVIATION` returns a valid `TapOptimizationResult`
- `all_tap_results` contains one row per tested tap position
- EV profiles are not included in tap optimization
- missing or invalid tap-position data raises `TapOptimizationError`

### N-1 tests

Test that:

- invalid outage line ID raises `InvalidLineOutageError`
- already disconnected outage line raises `InvalidLineOutageError`
- valid outage line returns the expected columns
- result index name is `alternative_line_id`
- no alternatives returns an empty table with the correct columns and index name

## Implementation Notes

- Keep Assignment 3 workflows independent, as required by the assignment. EV penetration should not affect tap optimization or N-1 unless explicitly passed into that workflow later.
- Prefer copying datasets before mutating line status or tap position so repeated analyses start from the original base grid.
- Reuse Assignment 2 aggregation logic instead of duplicating formulas.
- Reuse Assignment 1 `GraphProcessor` for downstream feeder membership and alternative-line lookup.
- Keep all public return values as pandas objects or small dataclasses so notebooks can display results easily.
- Do not commit the big demo dataset to git.
