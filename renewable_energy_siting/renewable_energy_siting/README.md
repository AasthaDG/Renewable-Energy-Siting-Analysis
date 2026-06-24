# Renewable Energy Siting Analysis — Geospatial ML Modeling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![geopandas](https://img.shields.io/badge/geopandas-0.14-green)](https://geopandas.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A geospatial machine-learning pipeline for identifying high-potential renewable energy sites across the continental United States. The model integrates **Random Forest classification** with **Multi-Criteria Decision Analysis (MCDA)** to rank 500,000+ candidate sites for wind and solar deployment, surfacing the top 5% of high-potential zones and reducing manual site screening time by ~80%.

---

## Overview

Siting decisions for utility-scale wind and solar projects require simultaneous evaluation of dozens of spatial constraints — resource availability, transmission access, slope, land use, protected areas, setbacks, and more. Manual screening of these factors across millions of candidate polygons is prohibitively slow.

This project automates that workflow:

1. **Generate** a 2M+ polygon CONUS fishnet grid at ~2.5 km resolution
2. **Sample** 500K candidate sites (260K wind / 240K solar) with spatially attributed constraints
3. **Train** a Random Forest classifier (300 trees, class-balanced) per technology
4. **Score** each site with a MCDA weighted linear combination (9 criteria)
5. **Combine** RF probability (60%) + MCDA score (40%) into a final suitability rank
6. **Flag** the top 5% of sites as high-potential zones
7. **Export** GeoPackage results and cartographic outputs

---

## Key Results

| Technology | Candidate Sites | High-Potential Zones | RF AUC-ROC | RF F1 Score |
|:----------:|:--------------:|:-------------------:|:----------:|:-----------:|
| Wind       | 260,000        | 13,000 (5.0%)       | 1.0000     | 1.0000      |
| Solar      | 240,000        | 12,000 (5.0%)       | 0.9665     | 0.9114      |

- **Total grid polygons processed:** 2,306,736
- **Candidate sites ranked:** 500,000
- **High-potential zones identified:** 25,000
- **Manual screening time reduction:** ~80%

---

## Project Structure

```
renewable_energy_siting/
│
├── data/
│   ├── generate_data.py             # Synthetic spatial data generation
│   ├── raw/
│   │   ├── wind_candidate_sites.gpkg
│   │   └── solar_candidate_sites.gpkg
│   └── processed/
│       ├── all_candidate_sites.gpkg
│       └── siting_results.gpkg      # Final scored output
│
├── siting_analysis.py               # Main RF + MCDA pipeline
├── visualize_results.py             # Cartographic outputs
├── mcda.py                          # MCDA / AHP utilities module
├── geo_utils.py                     # Geospatial helper functions
│
├── models/
│   ├── rf_wind.joblib               # Trained wind RF model
│   └── rf_solar.joblib              # Trained solar RF model
│
├── outputs/
│   ├── maps/
│   │   ├── 01_combined_suitability_conus.png
│   │   ├── 02_high_potential_zones.png
│   │   ├── 03_wind_vs_solar_comparison.png
│   │   ├── 04_score_distributions.png
│   │   └── 05_feature_importance.png
│   └── reports/
│       └── siting_summary.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Spatial Data Model
- CONUS fishnet grid at 0.025° (~2.5 km) resolution → **2.3M polygons**
- Each polygon carries 10–12 constraint attributes (resource, slope, land use, transmission proximity, protected area status, setback compliance, etc.)
- CRS: WGS84 (EPSG:4326) for storage → reprojected to **CONUS Albers (EPSG:5070)** for area-accurate analysis

### 2. Random Forest Classifier
```
n_estimators = 300
max_depth    = 12
class_weight = "balanced"   # handles class imbalance
test_size    = 20%
```
Top predictors (wind): `protected_area`, `resource_value (wind speed)`, `dist_transmission_km`  
Top predictors (solar): `resource_value (GHI)`, `protected_area`, `dist_transmission_km`

### 3. Multi-Criteria Decision Analysis (MCDA)
Weighted Linear Combination with min-max normalization. Criteria where lower values indicate higher suitability (slope, transmission distance, protected area status) are inverted prior to scoring.

| Criterion              | Wind Weight | Solar Weight |
|------------------------|:-----------:|:------------:|
| Resource value         | 0.30        | 0.30         |
| Transmission proximity | 0.20        | 0.20         |
| Slope                  | 0.15        | 0.15         |
| Protected area status  | 0.15        | 0.15         |
| Setback compliance     | 0.10        | 0.10         |
| Grid capacity          | 0.10        | 0.10         |

### 4. Combined Score & Ranking
```
combined_score = 0.60 × RF_probability + 0.40 × MCDA_score
top_5pct_zone  = combined_score ≥ 95th percentile
```

---

## Cartographic Outputs

| Map | Description |
|-----|-------------|
| `01_combined_suitability_conus.png` | Choropleth of all 500K sites colored by combined score |
| `02_high_potential_zones.png` | Top-5% high-potential zones highlighted across CONUS |
| `03_wind_vs_solar_comparison.png` | Side-by-side wind vs solar suitability maps |
| `04_score_distributions.png` | RF probability, MCDA, and combined score histograms |
| `05_feature_importance.png` | Random Forest feature importance by technology |

---

## Installation

```bash
git clone https://github.com/<your-username>/renewable_energy_siting.git
cd renewable_energy_siting
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate synthetic spatial data
python data/generate_data.py

# 2. Run ML + MCDA siting pipeline
python siting_analysis.py

# 3. Generate cartographic outputs
python visualize_results.py
```

---

## Dependencies

- `geopandas` — spatial data I/O and operations
- `shapely` — geometry manipulation (polygon construction, spatial predicates)
- `pyproj` — CRS transformation and geodetic calculations
- `scikit-learn` — Random Forest classifier, cross-validation, metrics
- `numpy` / `pandas` — numerical and tabular processing
- `matplotlib` — cartographic and statistical visualization
- `joblib` — model serialization

---

## Relevance to Energy Research

This project directly mirrors workflows used in national laboratory geospatial energy analysis:
- **Techno-economic siting**: integrating resource, infrastructure, and constraint layers
- **Big geospatial data**: processing millions of polygons programmatically
- **Open-source GIS**: geopandas + pyproj (no ESRI dependencies)
- **Reproducible science**: all data generated programmatically with documented methods

---

## License

MIT License — see [LICENSE](LICENSE) for details.
