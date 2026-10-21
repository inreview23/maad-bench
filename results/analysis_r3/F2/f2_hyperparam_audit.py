#!/usr/bin/env python3
"""f2_hyperparam_audit.py

Inspects and verifies the hyperparameters, seed mechanisms, and training/evaluation
pipelines of clip-level IForest and frame-level TempIF (TimeSeriesOD wrapping IForest).
"""

import inspect
import numpy as np
import pyod
from pyod.models.iforest import IForest
from pyod.models.ts_od import TimeSeriesOD, _DETECTOR_SHORTCUTS, resolve_detector


def audit():
    print("PyOD Version:", pyod.__version__)

    # 1. Base IForest signature and defaults
    sig_if = inspect.signature(IForest.__init__)
    print("\n[Base IForest __init__ parameters]")
    for p, param in sig_if.parameters.items():
        if p != "self":
            print(f"  {p}: default={param.default}")

    # 2. TimeSeriesOD signature and defaults
    sig_tsod = inspect.signature(TimeSeriesOD.__init__)
    print("\n[TimeSeriesOD __init__ parameters]")
    for p, param in sig_tsod.parameters.items():
        if p != "self":
            print(f"  {p}: default={param.default}")

    # 3. How TimeSeriesOD instantiates IForest
    shortcut = _DETECTOR_SHORTCUTS.get("IForest")
    print(f"\n_DETECTOR_SHORTCUTS['IForest']: {shortcut}")

    # 4. Resolve detector under TimeSeriesOD
    resolved = resolve_detector("IForest", contamination=0.1)
    print(f"Resolved detector type: {type(resolved)}")
    print(f"Resolved detector params: n_estimators={resolved.n_estimators}, "
          f"max_samples={resolved.max_samples}, contamination={resolved.contamination}, "
          f"random_state={resolved.random_state}")

    # 5. Check temporal.py invocation:
    # TimeSeriesOD(detector="IForest", window_size=16, step=4, score_aggregation="max")
    ts_model = TimeSeriesOD(detector="IForest", window_size=16, step=4, score_aggregation="max")
    print(f"\nTempIF instantiation parameters:")
    print(f"  detector: {ts_model.detector}")
    print(f"  window_size: {ts_model.window_size}")
    print(f"  step: {ts_model.step}")
    print(f"  score_aggregation: {ts_model.score_aggregation}")
    print(f"  contamination: {ts_model.contamination}")


if __name__ == "__main__":
    audit()
