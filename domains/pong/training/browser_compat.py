"""Float32 browser-Actor compatibility checks used at Pong export time."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def browser_style_probabilities(model: Mapping[str, Any], features: Mapping[str, float]) -> np.ndarray:
    """Evaluate the JSON actor with the same layer order as ``web/app.js``."""
    names = tuple(model["signature"]["feature_names"])
    missing = [name for name in names if name not in features]
    if missing:
        raise ValueError(f"browser observation is missing: {missing[:3]}")
    values = np.asarray([features[name] for name in names], dtype=np.float32)
    for layer in model["layers"]:
        weights = np.asarray(layer["weight"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        # Mirror JavaScript's row.reduce form.  It also avoids platform BLAS
        # floating-point warnings on otherwise finite small vectors.
        values = np.sum(weights * values[None, :], axis=1, dtype=np.float32) + bias
        if layer["activation"] == "tanh":
            values = np.tanh(values).astype(np.float32)
    shifted = values - np.max(values)
    probabilities = np.exp(shifted, dtype=np.float32)
    return probabilities / probabilities.sum(dtype=np.float32)


def verify_browser_export(model: Mapping[str, Any], features: Mapping[str, float],
                          python_probabilities: np.ndarray) -> dict[str, Any]:
    browser = browser_style_probabilities(model, features)
    python = np.asarray(python_probabilities, dtype=np.float32)
    maximum_error = float(np.max(np.abs(browser - python)))
    browser_action = int(np.argmax(browser))
    python_action = int(np.argmax(python))
    if maximum_error > 1e-5 or browser_action != python_action:
        raise RuntimeError(
            "Python and browser Actor outputs disagree: "
            f"max_error={maximum_error:.8f}, python_action={python_action}, browser_action={browser_action}"
        )
    return {"maximum_probability_error": maximum_error, "python_action_index": python_action,
            "browser_action_index": browser_action, "deterministic_action_matches": True}
