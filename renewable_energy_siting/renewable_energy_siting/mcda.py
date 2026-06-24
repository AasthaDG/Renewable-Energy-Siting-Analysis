"""
mcda.py
Multi-Criteria Decision Analysis (MCDA) utilities for renewable energy siting.

Implements weighted linear combination (WLC) and optional AHP weight derivation.
All criteria are normalized to [0, 1] before weighting; criteria where lower
raw values indicate higher suitability are inverted automatically.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional


INVERT_CRITERIA = {
    "dist_transmission_km",
    "dist_road_km",
    "slope_pct",
    "protected_area",
    "population_density",
    "cloud_cover_pct",
    "noise_sensitive_area",
}


def normalize_criteria(
    df: pd.DataFrame,
    criteria: List[str],
    invert: Optional[set] = None
) -> pd.DataFrame:
    """
    Min-max normalize each criterion column.
    Columns in `invert` are flipped so that 1 = most suitable.

    Parameters
    ----------
    df       : DataFrame with criterion columns
    criteria : list of column names to normalize
    invert   : set of column names to invert after normalization

    Returns
    -------
    DataFrame of normalized scores, same index as df
    """
    if invert is None:
        invert = INVERT_CRITERIA

    available = [c for c in criteria if c in df.columns]
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(df[available])
    normed_df = pd.DataFrame(normed, columns=available, index=df.index)

    for col in available:
        if col in invert:
            normed_df[col] = 1.0 - normed_df[col]

    return normed_df


def weighted_linear_combination(
    normed_df: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.Series:
    """
    Compute WLC suitability score.

    Parameters
    ----------
    normed_df : normalized criteria DataFrame
    weights   : {criterion_name: weight}  (need not sum to 1; auto-normalized)

    Returns
    -------
    Series of suitability scores in [0, 1]
    """
    available = [c for c in weights if c in normed_df.columns]
    total_w   = sum(weights[c] for c in available)

    if total_w == 0:
        raise ValueError("All weight keys are missing from the DataFrame.")

    score = sum(normed_df[c] * (weights[c] / total_w) for c in available)
    return score


def ahp_weights_from_matrix(comparison_matrix: np.ndarray) -> np.ndarray:
    """
    Derive priority weights from a pairwise comparison matrix (AHP).

    Parameters
    ----------
    comparison_matrix : n×n numpy array (Saaty scale 1–9)

    Returns
    -------
    Normalized weight vector (sums to 1.0) and consistency ratio (CR).
    Raises ValueError if CR > 0.10 (matrix is inconsistent).
    """
    n = comparison_matrix.shape[0]
    # Column normalization
    col_sums = comparison_matrix.sum(axis=0)
    normed   = comparison_matrix / col_sums
    weights  = normed.mean(axis=1)

    # Consistency check
    lam_max = (comparison_matrix @ weights / weights).mean()
    ci      = (lam_max - n) / (n - 1)
    ri_map  = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12,
               6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri  = ri_map.get(n, 1.49)
    cr  = ci / ri if ri > 0 else 0.0

    if cr > 0.10:
        raise ValueError(
            f"AHP comparison matrix is inconsistent (CR={cr:.3f} > 0.10). "
            "Revise pairwise comparisons."
        )
    return weights, cr


def compute_mcda(
    df: pd.DataFrame,
    weights: Dict[str, float],
    invert: Optional[set] = None
) -> pd.Series:
    """
    Full MCDA pipeline: normalize → WLC → score series.

    Parameters
    ----------
    df      : raw attribute DataFrame
    weights : criterion → weight mapping
    invert  : criteria to invert (defaults to module-level INVERT_CRITERIA)

    Returns
    -------
    Suitability score Series in [0, 1]
    """
    criteria  = list(weights.keys())
    normed_df = normalize_criteria(df, criteria, invert)
    return weighted_linear_combination(normed_df, weights)


# ── Example AHP matrix for wind siting (6 criteria) ─────────────────────────
# Criteria: resource_value, dist_transmission, slope, protected_area,
#           setback_met, grid_capacity
WIND_AHP_MATRIX = np.array([
    [1,   3,   5,   3,   7,   3  ],   # resource_value
    [1/3, 1,   3,   1,   5,   1  ],   # dist_transmission
    [1/5, 1/3, 1,   1/3, 3,   1/3],   # slope
    [1/3, 1,   3,   1,   5,   1  ],   # protected_area
    [1/7, 1/5, 1/3, 1/5, 1,   1/5],   # setback_met
    [1/3, 1,   3,   1,   5,   1  ],   # grid_capacity
])


if __name__ == "__main__":
    print("=== MCDA Module Self-Test ===\n")

    # Derive AHP weights
    weights_ahp, cr = ahp_weights_from_matrix(WIND_AHP_MATRIX)
    criteria_names = [
        "resource_value", "dist_transmission_km", "slope_pct",
        "protected_area", "setback_met", "grid_capacity_mw"
    ]
    print("AHP-derived weights (wind siting):")
    for name, w in zip(criteria_names, weights_ahp):
        print(f"  {name:30s}: {w:.4f}")
    print(f"  Consistency Ratio (CR): {cr:.4f}  ({'OK' if cr<0.10 else 'FAIL'})\n")

    # Dummy data
    np.random.seed(0)
    n = 1000
    df_test = pd.DataFrame({
        "resource_value":        np.random.normal(7, 2, n).clip(2, 14),
        "dist_transmission_km":  np.abs(np.random.exponential(25, n)),
        "slope_pct":             np.abs(np.random.normal(5, 8, n)),
        "protected_area":        np.random.choice([0,1], n, p=[0.78, 0.22]),
        "setback_met":           np.random.choice([0,1], n, p=[0.15, 0.85]),
        "grid_capacity_mw":      np.random.normal(150, 80, n).clip(10, 500),
    })

    weights_dict = dict(zip(criteria_names, weights_ahp))
    scores = compute_mcda(df_test, weights_dict)
    print(f"MCDA scores — mean: {scores.mean():.4f}, std: {scores.std():.4f}")
    print(f"Top-5% threshold: {scores.quantile(0.95):.4f}")
    print("\nSelf-test PASSED ✓")
