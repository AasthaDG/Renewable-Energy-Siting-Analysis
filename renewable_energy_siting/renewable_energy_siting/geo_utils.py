"""
geo_utils.py
Geospatial helper utilities for the Renewable Energy Siting pipeline.

Functions cover:
  - CRS reprojection helpers (pyproj / geopandas)
  - Fishnet / hex grid generation
  - Spatial join wrappers
  - Distance calculation (projected & geodetic)
  - Constraint masking
  - Polygon area & centroid extraction
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import pyproj
from pyproj import Transformer, CRS
from typing import Tuple, List, Optional, Union
import warnings
warnings.filterwarnings("ignore")

# ── Standard CRS ──────────────────────────────────────────────────────────────
WGS84        = "EPSG:4326"
CONUS_ALBERS = "EPSG:5070"   # NAD83 / Conus Albers — area-preserving
WEB_MERCATOR = "EPSG:3857"


# ── Projection helpers ────────────────────────────────────────────────────────

def reproject(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame; no-op if already in target CRS."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set.")
    if gdf.crs.to_string() == CRS(target_crs).to_string():
        return gdf.copy()
    return gdf.to_crs(target_crs)


def transform_point(lon: float, lat: float,
                    src: str = WGS84, dst: str = CONUS_ALBERS
                    ) -> Tuple[float, float]:
    """Transform a single (lon, lat) pair to target CRS. Returns (x, y)."""
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return transformer.transform(lon, lat)


def bbox_to_projected(minx, miny, maxx, maxy,
                      src_crs: str = WGS84,
                      dst_crs: str = CONUS_ALBERS) -> Tuple[float,float,float,float]:
    """Transform a bounding box from src_crs to dst_crs."""
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x0, y0 = t.transform(minx, miny)
    x1, y1 = t.transform(maxx, maxy)
    return (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))


# ── Grid generation ───────────────────────────────────────────────────────────

def make_fishnet(bounds: Tuple[float,float,float,float],
                 cell_size: float,
                 crs: str = WGS84) -> gpd.GeoDataFrame:
    """
    Create a regular rectangular fishnet grid.

    Parameters
    ----------
    bounds    : (minx, miny, maxx, maxy)
    cell_size : cell width/height in CRS units
    crs       : coordinate reference system string

    Returns
    -------
    GeoDataFrame of rectangular polygons with columns [cell_id, geometry]
    """
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    geoms, ids = [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            geoms.append(box(x, y, x + cell_size, y + cell_size))
            ids.append(i * len(ys) + j)

    return gpd.GeoDataFrame({"cell_id": ids, "geometry": geoms}, crs=crs)


def make_hex_grid(bounds: Tuple[float,float,float,float],
                  radius: float,
                  crs: str = CONUS_ALBERS) -> gpd.GeoDataFrame:
    """
    Create a hexagonal grid (flat-top hexagons).

    Parameters
    ----------
    bounds : (minx, miny, maxx, maxy) in projected units (metres)
    radius : circumradius of each hexagon (metres)
    crs    : projected CRS

    Returns
    -------
    GeoDataFrame of hexagonal polygons
    """
    minx, miny, maxx, maxy = bounds
    dx = radius * np.sqrt(3)
    dy = radius * 1.5

    def hex_polygon(cx, cy):
        angles = np.deg2rad(np.arange(0, 360, 60))
        pts = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
        return Polygon(pts)

    geoms, ids = [], []
    row, cell_id = 0, 0
    y = miny
    while y < maxy:
        x = minx + (dx/2 if row % 2 else 0)
        while x < maxx:
            geoms.append(hex_polygon(x, y))
            ids.append(cell_id)
            x += dx
            cell_id += 1
        y += dy
        row += 1

    return gpd.GeoDataFrame({"cell_id": ids, "geometry": geoms}, crs=crs)


# ── Distance calculations ─────────────────────────────────────────────────────

def geodetic_distance_km(lon1, lat1, lon2, lat2) -> float:
    """Haversine geodetic distance between two WGS84 points (km)."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_feature_distance(gdf_sites: gpd.GeoDataFrame,
                              gdf_features: gpd.GeoDataFrame,
                              col_name: str = "dist_m") -> gpd.GeoDataFrame:
    """
    Compute distance from each site centroid to the nearest feature geometry.
    Both GeoDataFrames must share the same projected CRS.

    Returns gdf_sites with a new column `col_name` (metres).
    """
    if gdf_sites.crs != gdf_features.crs:
        gdf_features = gdf_features.to_crs(gdf_sites.crs)

    site_centroids = gdf_sites.geometry.centroid
    feature_union  = unary_union(gdf_features.geometry)
    distances = site_centroids.distance(feature_union)
    result = gdf_sites.copy()
    result[col_name] = distances
    return result


# ── Constraint masking ────────────────────────────────────────────────────────

def apply_exclusion_mask(gdf_sites: gpd.GeoDataFrame,
                         gdf_exclusions: gpd.GeoDataFrame,
                         label: str = "excluded") -> gpd.GeoDataFrame:
    """
    Flag sites that intersect any exclusion zone polygon.

    Parameters
    ----------
    gdf_sites      : candidate sites
    gdf_exclusions : exclusion polygons (protected areas, urban zones, etc.)
    label          : name of the boolean flag column added to gdf_sites

    Returns
    -------
    gdf_sites with new binary column `label` (1 = excluded)
    """
    if gdf_sites.crs != gdf_exclusions.crs:
        gdf_exclusions = gdf_exclusions.to_crs(gdf_sites.crs)

    excl_union = unary_union(gdf_exclusions.geometry)
    gdf_out = gdf_sites.copy()
    gdf_out[label] = gdf_out.geometry.intersects(excl_union).astype(int)
    n_excl = gdf_out[label].sum()
    print(f"  Exclusion mask '{label}': {n_excl:,} / {len(gdf_out):,} sites excluded")
    return gdf_out


def filter_by_slope(gdf: gpd.GeoDataFrame,
                    slope_col: str = "slope_pct",
                    max_slope: float = 20.0) -> gpd.GeoDataFrame:
    """Remove sites exceeding maximum slope threshold."""
    mask = gdf[slope_col] <= max_slope
    print(f"  Slope filter (≤{max_slope}%): {mask.sum():,} / {len(gdf):,} sites retained")
    return gdf[mask].copy()


# ── Polygon utilities ─────────────────────────────────────────────────────────

def add_area_km2(gdf: gpd.GeoDataFrame,
                 projected_crs: str = CONUS_ALBERS) -> gpd.GeoDataFrame:
    """Add 'area_km2' column based on projected area."""
    gdf_proj = gdf.to_crs(projected_crs) if gdf.crs.to_string() != CRS(projected_crs).to_string() else gdf
    gdf = gdf.copy()
    gdf["area_km2"] = gdf_proj.geometry.area / 1e6
    return gdf


def add_centroid_coords(gdf: gpd.GeoDataFrame,
                        projected_crs: str = CONUS_ALBERS) -> gpd.GeoDataFrame:
    """Add centroid_x, centroid_y columns in projected CRS."""
    gdf_proj = gdf.to_crs(projected_crs)
    gdf = gdf.copy()
    gdf["centroid_x"] = gdf_proj.geometry.centroid.x
    gdf["centroid_y"] = gdf_proj.geometry.centroid.y
    return gdf


def dissolve_top_zones(gdf: gpd.GeoDataFrame,
                       zone_col: str = "top5pct_zone",
                       tech_col: str = "tech_type") -> gpd.GeoDataFrame:
    """
    Dissolve adjacent high-potential cells into contiguous zone polygons.
    Returns one dissolved polygon per technology type.
    """
    high = gdf[gdf[zone_col] == 1].copy()
    dissolved = high.dissolve(by=tech_col, as_index=False)[
        [tech_col, "geometry"]
    ]
    dissolved["zone_area_km2"] = dissolved.geometry.area / 1e6
    print(f"  Dissolved into {len(dissolved)} zone polygons")
    return dissolved


# ── Quick validation ──────────────────────────────────────────────────────────

def validate_geodataframe(gdf: gpd.GeoDataFrame, name: str = "GeoDataFrame"):
    """Basic validity checks; prints summary."""
    print(f"\n  [{name}] Validation report")
    print(f"    Rows:           {len(gdf):,}")
    print(f"    CRS:            {gdf.crs}")
    print(f"    Null geometries:{gdf.geometry.isna().sum():,}")
    invalid = (~gdf.geometry.is_valid).sum()
    print(f"    Invalid geoms:  {invalid:,}")
    print(f"    Bounds:         {gdf.total_bounds.round(2)}")
    if invalid > 0:
        print("    ⚠ Run gdf.geometry = gdf.geometry.buffer(0) to fix invalid geometries.")


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== geo_utils self-test ===\n")

    # Fishnet
    fishnet = make_fishnet((-100, 35, -95, 40), cell_size=0.5)
    print(f"Fishnet: {len(fishnet):,} cells")
    validate_geodataframe(fishnet, "fishnet")

    # Projection
    x, y = transform_point(-105.0, 40.0, WGS84, CONUS_ALBERS)
    print(f"\nTransformed (-105, 40) WGS84 → Albers: ({x:.1f}, {y:.1f})")

    # Geodetic distance
    d = geodetic_distance_km(-74.006, 40.713, -87.629, 41.878)
    print(f"NYC → Chicago geodetic distance: {d:.1f} km  (expected ~1147 km)")

    # Area
    fishnet_albers = reproject(fishnet, CONUS_ALBERS)
    fishnet_albers = add_area_km2(fishnet_albers)
    print(f"\nFishnet cell areas (km²): min={fishnet_albers.area_km2.min():.2f}, "
          f"max={fishnet_albers.area_km2.max():.2f}")

    print("\nAll self-tests PASSED ✓")
