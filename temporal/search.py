"""Builds the list of configurations evaluated in the dev search (Package 4, step 4).

Every configuration is a dict ``{"config_id", "condition", "params"}``. IDs are
deterministic (same YAML + same seed -> same list), so a CSV row can always be
traced back to its exact parameters.
"""

from __future__ import annotations

import numpy as np
import yaml


def load_search_space(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _theta_grid(spec: dict) -> list[float]:
    # Rounded so that 0.15 is stored as 0.15, not 0.15000000000000002.
    n = int(round((spec["stop"] - spec["start"]) / spec["step"])) + 1
    return [round(spec["start"] + i * spec["step"], 4) for i in range(n)]


def t1_configs(space: dict) -> list[dict]:
    """4 windows x 19 thresholds = 76 configurations."""
    out = []
    for w in space["t1"]["windows_s"]:
        for theta in _theta_grid(space["t1"]["theta"]):
            out.append({"config_id": f"T1_w{w:g}_th{theta:.2f}", "condition": "T1",
                        "params": {"window_s": float(w), "theta": theta,
                                   "min_area_frac": space["min_area_frac"]}})
    return out


def t2_configs(space: dict) -> list[dict]:
    """``n_configs`` random TCA configurations drawn with a fixed seed.

    tau is log-uniform because its useful range spans more than an order of
    magnitude (0.1-3 s); the other ranges are narrow enough for uniform draws.
    """
    s = space["t2"]
    rng = np.random.default_rng(space["seed"])
    out = []
    for i in range(s["n_configs"]):
        tau = float(np.exp(rng.uniform(np.log(s["tau"][0]), np.log(s["tau"][1]))))
        high = float(rng.uniform(*s["high"]))
        low = float(rng.uniform(*s["low"]))
        while low >= high:                 # hysteresis needs low < high
            low = float(rng.uniform(*s["low"]))
        params = {
            "tau": round(tau, 4),
            "present_dur": round(float(rng.uniform(*s["present_dur"])), 4),
            "absence_dur": round(float(rng.uniform(*s["absence_dur"])), 4),
            "timeout": round(float(rng.uniform(*s["timeout"])), 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "iou_thr": s["iou_thr"],
            "sigma_px": s["sigma_px"],
            "min_area_frac": space["min_area_frac"],
        }
        out.append({"config_id": f"T2_{i:02d}", "condition": "T2", "params": params})
    return out


def t0_configs(space: dict) -> list[dict]:
    """T0 tunes its own confidence threshold on dev (17 values by default)."""
    return [t0_config(c, space) for c in _theta_grid(space["t0"]["conf"])]


def t0_config(conf_thr: float, space: dict) -> dict:
    """One T0 configuration with a given confidence threshold."""
    return {"config_id": f"T0_conf{conf_thr:.2f}", "condition": "T0",
            "params": {"conf_thr": float(conf_thr),
                       "min_area_frac": space["min_area_frac"]}}
