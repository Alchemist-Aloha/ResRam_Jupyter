# Qt Frontend Responsiveness Design

## Goal

Improve the ResRAM Qt frontend so it remains responsive while recalculating plots and fitting parameters. The current layout should stay familiar: plots remain on the left, controls and the parameter table remain on the right. The focus is robustness and performance, not a broad visual redesign.

## Current Context

The GUI is implemented in `src/resram_ng/ResRamQt.py` with PyQt6 and pyqtgraph. It already uses `QThreadPool`, `QRunnable` workers, debounced table edits, and cached plot item references. The app can still feel unresponsive because recalculation requests can overlap, workers read mutable GUI state, and old calculation results can arrive after newer edits or folder loads.

## Recommended Approach

Use a small calculation coordination layer inside the existing `SpectrumApp` instead of changing the application architecture. Each plot recalculation gets a monotonically increasing request id. The GUI allows at most one active plot calculation and tracks whether another calculation is pending. If the user edits parameters while a calculation is running, the app records the pending update and starts exactly one follow-up calculation after the active worker finishes.

This preserves the current UI and is lower risk than a full state-management rewrite. It directly targets responsiveness by bounding background work and preventing stale results from updating the plots.

## Responsiveness Model

Table edits should not start unlimited concurrent calculations. A debounced edit starts a coordinated calculation request. If no calculation is running, the request starts immediately. If a calculation is running, the app marks a pending request and updates the status label to show that a follow-up calculation is queued.

Each calculation result includes the request id that created it. The GUI applies a result only when it matches the latest active request and the dataset shape is still compatible with the current UI state. Older results are ignored. Folder loads and initialization bump the request id and clear cached plot items, so results from the previous dataset cannot overwrite the new plots.

The user should see a compact status indicator with states such as `Ready`, `Calculating...`, `Update pending`, and `Calculation failed`. Calculation errors should not leave the app in a permanently busy state.

## Frontend Behavior

The main layout remains unchanged. A small status row or label is added near the existing buttons. The parameter table stays editable while plot calculations run, because edits are cheap and can be coalesced into a pending recalculation. The fit button remains disabled while fitting. Ordinary plot recalculations should not pile up during a fit.

Folder loading keeps the current message-box behavior for invalid folders. Successful folder loads reset table state, clear plot caches, invalidate old calculation requests, refresh the directory label, and start one coordinated plot calculation for the new dataset.

## Performance Changes

Plot workers should operate on a calculation snapshot rather than a mutable `obj_load` reference that the UI continues to edit. The snapshot can initially be a deep copy of the loaded object if that is the safest local change. If copying is too expensive or incompatible, use a focused snapshot containing the arrays and scalar parameters needed by `cross_sections` and Raman spectrum generation.

The plot update path should prepare repeated arrays once per result, including the real-valued Raman excitation profile matrix, instead of recomputing transposes inside per-line update loops. Existing pyqtgraph items should be reused when the number of pumps, modes, and axes are unchanged. Plot items should be rebuilt only after dataset shape changes, visible-series structure changes, or folder loads.

Defensive shape checks should discard incompatible results before touching plot items. Discarded results should not be treated as user-facing failures when they are caused by legitimate newer edits or folder loads.

## Error Handling

Worker exceptions are reported through the existing Qt signal path. The GUI records the failure in the status label, clears the active calculation flag, and starts a pending calculation if one was requested after the failed worker began. Folder load failures continue to use `QMessageBox.warning` because they require user attention.

Unexpected GUI-level exceptions continue to use the existing global exception hook.

## Testing and Verification

Add focused tests for coordination behavior where practical without requiring a visible GUI. The key cases are: a second request during an active calculation becomes pending, stale results are ignored, failed workers clear the busy state, and folder loads invalidate old results.

Run the existing test suite with `uv run pytest`. If the environment can launch Qt, run a GUI smoke check with the default example dataset. If Qt cannot launch in the current environment, document that limitation and rely on import/startup tests plus unit coverage.

## Out of Scope

This pass does not replace PyQt6, replace pyqtgraph, redesign the entire layout, or change the ResRAM numerical model. It also does not optimize the Rust or Python scientific kernels except where frontend data flow currently causes unnecessary UI work.
