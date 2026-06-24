"""
generate_data.py
Generates synthetic geospatial datasets for renewable energy siting analysis.
Produces 2M+ spatial polygons representing wind and solar candidate sites
across the continental US with associated constraint attributes.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union
import pyproj
from pyproj import Transformer
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── CONUS bounding box (lon/lat WGS84) ──────────────────────────────────────
CONUS_BOUNDS = (-124.8, 24.5, -66.9, 49.4)
GRID_STEP    = 0.025          # ~2.5 km cells → ~2 M polygons across CONUS

def generate_grid_polygons(bounds, step):
    """Create a regular fishnet grid of rectangular polygons."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    print(f"  Grid dimensions: {len(xs)} cols × {len(ys)} rows = {len(xs)*len(ys):,} cells")

    geoms, ids = [], []
    cell_id = 0
    for x in xs:
        for y in ys:
            geoms.append(box(x, y, x + step, y + step))
            ids.append(cell_id)
            cell_id += 1
    return geoms, ids

def assign_wind_attributes(n):
    """Simulate wind-resource and constraint attributes."""
    return {
        "wind_speed_ms":      np.random.normal(7.2, 2.1, n).clip(2, 14),
        "slope_pct":          np.abs(np.random.normal(5, 8, n)).clip(0, 60),
        "dist_road_km":       np.abs(np.random.exponential(12, n)).clip(0.1, 200),
        "dist_transmission_km": np.abs(np.random.exponential(25, n)).clip(0.1, 300),
        "protected_area":     np.random.choice([0, 1], n, p=[0.78, 0.22]),
        "land_use_code":      np.random.choice([1,2,3,4,5,6], n,
                                               p=[0.30,0.25,0.20,0.12,0.08,0.05]),
        "elevation_m":        np.random.normal(600, 400, n).clip(0, 4000),
        "population_density": np.abs(np.random.exponential(50, n)),
        "setback_met":        np.random.choice([0, 1], n, p=[0.15, 0.85]),
        "grid_capacity_mw":   np.random.normal(150, 80, n).clip(10, 500),
        "noise_sensitive_area": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    }

def assign_solar_attributes(n):
    """Simulate solar-resource and constraint attributes."""
    return {
        "ghi_kwh_m2_day":     np.random.normal(5.1, 1.2, n).clip(2.5, 8.0),
        "slope_pct":          np.abs(np.random.normal(4, 6, n)).clip(0, 45),
        "dist_road_km":       np.abs(np.random.exponential(10, n)).clip(0.1, 200),
        "dist_transmission_km": np.abs(np.random.exponential(20, n)).clip(0.1, 300),
        "protected_area":     np.random.choice([0, 1], n, p=[0.78, 0.22]),
        "land_use_code":      np.random.choice([1,2,3,4,5,6], n,
                                               p=[0.30,0.25,0.20,0.12,0.08,0.05]),
        "elevation_m":        np.random.normal(500, 350, n).clip(0, 3500),
        "population_density": np.abs(np.random.exponential(50, n)),
        "setback_met":        np.random.choice([0, 1], n, p=[0.10, 0.90]),
        "grid_capacity_mw":   np.random.normal(120, 70, n).clip(10, 400),
        "cloud_cover_pct":    np.random.normal(35, 15, n).clip(5, 80),
        "aspect_degrees":     np.random.uniform(0, 360, n),
    }

def compute_wind_label(df):
    """Rule-based suitability label for wind (used as RF training target)."""
    score = (
        (df["wind_speed_ms"] > 7.0).astype(int) * 3 +
        (df["slope_pct"]     < 15).astype(int)  * 2 +
        (df["dist_transmission_km"] < 30).astype(int) * 2 +
        (df["protected_area"] == 0).astype(int)  * 3 +
        (df["setback_met"]   == 1).astype(int)   * 2 +
        (df["land_use_code"].isin([1,2,3])).astype(int) * 1 +
        (df["population_density"] < 100).astype(int) * 1
    )
    return (score >= 9).astype(int)   # top ~5 % will be "suitable"

def compute_solar_label(df):
    """Rule-based suitability label for solar."""
    score = (
        (df["ghi_kwh_m2_day"] > 5.5).astype(int)       * 3 +
        (df["slope_pct"]      < 10).astype(int)          * 2 +
        (df["dist_transmission_km"] < 25).astype(int)    * 2 +
        (df["protected_area"] == 0).astype(int)           * 3 +
        (df["setback_met"]   == 1).astype(int)            * 2 +
        (df["cloud_cover_pct"] < 30).astype(int)          * 2 +
        (df["land_use_code"].isin([1,2,3])).astype(int)   * 1
    )
    return (score >= 10).astype(int)

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Generating synthetic geospatial dataset ===\n")

    # 1. Build grid
    print("[1/4] Building CONUS fishnet grid …")
    geoms, ids = generate_grid_polygons(CONUS_BOUNDS, GRID_STEP)
    total_cells = len(geoms)
    print(f"  Total grid cells: {total_cells:,}\n")

    # 2. Sample 500 K candidate sites (wind + solar split)
    print("[2/4] Sampling 500 K candidate sites …")
    N_WIND  = 260_000
    N_SOLAR = 240_000
    N_TOTAL = N_WIND + N_SOLAR

    idx_wind  = np.random.choice(total_cells, N_WIND,  replace=False)
    idx_solar = np.random.choice(total_cells, N_SOLAR, replace=False)

    # Wind GeoDataFrame
    wind_attrs = assign_wind_attributes(N_WIND)
    wind_attrs["cell_id"]   = idx_wind
    wind_attrs["tech_type"] = "wind"
    gdf_wind = gpd.GeoDataFrame(
        wind_attrs,
        geometry=[geoms[i] for i in idx_wind],
        crs="EPSG:4326"
    )
    gdf_wind["suitable"] = compute_wind_label(gdf_wind)

    # Solar GeoDataFrame
    solar_attrs = assign_solar_attributes(N_SOLAR)
    solar_attrs["cell_id"]   = idx_solar
    solar_attrs["tech_type"] = "solar"
    gdf_solar = gpd.GeoDataFrame(
        solar_attrs,
        geometry=[geoms[i] for i in idx_solar],
        crs="EPSG:4326"
    )
    gdf_solar["suitable"] = compute_solar_label(gdf_solar)

    print(f"  Wind sites:  {N_WIND:,}  (suitable: {gdf_wind['suitable'].sum():,})")
    print(f"  Solar sites: {N_SOLAR:,} (suitable: {gdf_solar['suitable'].sum():,})\n")

    # 3. Save raw
    print("[3/4] Saving raw GeoPackage files …")
    gdf_wind.to_file("data/raw/wind_candidate_sites.gpkg", driver="GPKG")
    gdf_solar.to_file("data/raw/solar_candidate_sites.gpkg", driver="GPKG")
    print("  Saved: data/raw/wind_candidate_sites.gpkg")
    print("  Saved: data/raw/solar_candidate_sites.gpkg\n")

    # 4. Combined processed
    print("[4/4] Building combined processed dataset …")
    common_cols = ["cell_id", "tech_type", "slope_pct", "dist_road_km",
                   "dist_transmission_km", "protected_area", "land_use_code",
                   "elevation_m", "population_density", "setback_met",
                   "grid_capacity_mw", "suitable", "geometry"]

    gdf_wind_c  = gdf_wind.rename(columns={"wind_speed_ms":"resource_value"})[
        ["cell_id","tech_type","resource_value","slope_pct","dist_road_km",
         "dist_transmission_km","protected_area","land_use_code","elevation_m",
         "population_density","setback_met","grid_capacity_mw","suitable","geometry"]]

    gdf_solar_c = gdf_solar.rename(columns={"ghi_kwh_m2_day":"resource_value"})[
        ["cell_id","tech_type","resource_value","slope_pct","dist_road_km",
         "dist_transmission_km","protected_area","land_use_code","elevation_m",
         "population_density","setback_met","grid_capacity_mw","suitable","geometry"]]

    gdf_all = pd.concat([gdf_wind_c, gdf_solar_c], ignore_index=True)
    gdf_all = gpd.GeoDataFrame(gdf_all, crs="EPSG:4326")
    gdf_all.to_file("data/processed/all_candidate_sites.gpkg", driver="GPKG")
    print(f"  Saved: data/processed/all_candidate_sites.gpkg  ({len(gdf_all):,} rows)\n")

    print("=== Data generation complete ===")
    print(f"  Total polygons in grid:        {total_cells:,}")
    print(f"  Total candidate sites sampled: {N_TOTAL:,}")
    print(f"  Suitable sites (wind):         {gdf_wind['suitable'].sum():,}")
    print(f"  Suitable sites (solar):        {gdf_solar['suitable'].sum():,}")
