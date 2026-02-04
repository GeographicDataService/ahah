# AHAH v5 — Access to Healthy Assets and Hazards

[Please see the relevant tags in this repo to access the documentation for v3 and v4.]

A Python toolkit for calculating the Access to Healthy Assets and Hazards (AHAH) index, measuring environmental factors that influence health outcomes at small-area level across Great Britain.

## Overview

The AHAH index quantifies spatial accessibility to health-promoting and health-damaging features of the built environment. This repository provides the complete pipeline for calculating AHAH v5, including:

- **High-performance routing infrastructure** using a Valhalla cluster
- **Scalable accessibility calculations** from 1.7M+ postcodes to points of interest
- **Composite index construction** following the established AHAH methodology

The index combines four domains:
- **Health Services** — GP surgeries, pharmacies, hospitals, dentists, leisure centres
- **Green/Blue Space** — Greenspace (Passive / NDVI), green and bluespace accessibility
- **Air Quality** — NO₂, PM₁₀, SO₂ concentrations
- **Retail Hazards** — Fast food outlets, gambling venues, pubs/bars, vape / tobacco retailers

## Repository Structure

```
├── code/ # Core AHAH calculation modules
│   ├── __init__.py
│   ├── valhalla_cluster.py      # Docker-based routing cluster management
│   └── poi_accessibility.py     # Parallel POI accessibility calculations
├── AHAH_Data_Setup.ipynb        # Data acquisition and preparation
├── AHAH_Routing_Cluster_Setup.ipynb  # Valhalla cluster configuration
├── AHAH_Access_Calculations.ipynb    # Accessibility metric generation
├── AHAH_Build_Indicator.ipynb   # Final index construction
├── requirements.txt
└── README.md
```
Additionally - ```vegetation_index.ipynb``` creates NDVI greenspace via GEE. These outputs are integrated into AHAH and also our Greenspace [Supplementary Indicators](https://data.geods.ac.uk/dataset/small-area-uk-vegetation-indices)).

### Repo Data Directory Structure

```
data/
├── airquality/        # Air quality surfaces
├── boundary/          # Census boundary geometries
├── green_blue/         # NDVI greenspace data
├── health/              # Health services datasets
├── postcodes/          # Postcode coordinates with LSOA linkage

The following are not included in the repo but inetgral to the workflow:

├── raw_data/          # Original downloaded datasets - these are not included in the repo and must be downloaded separately
├── retail/          # Retail data not included in the repo given commercial licensing
├── routing_results/    # Intermediate routing results
├── valhalla_tiles/     # Valhalla routing tiles directory

```


## Installation

### Prerequisites

- Python 3.10+
- Docker (for Valhalla routing cluster)
- ~6GB disk space for UK routing tiles - there will be a new root directory generated called 'valhalla_tiles'

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ahah.git
cd ahah

# Create environment and install dependencies
pip install -r requirements.txt
```

### Dependencies

```
# Core
docker>=6.0.0
pyyaml>=6.0
requests>=2.28.0

# Geospatial
geopandas>=0.14.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pyarrow>=14.0.0

# Progress
tqdm>=4.65.0
```

## Workflow

The AHAH calculation follows four sequential stages:

### 1. Data Setup (`AHAH_Data_Setup.ipynb`)

Acquires and prepares all input datasets:
- Census boundary geometries (LSOA/Data Zone for 2021/22)
- Postcode coordinates with LSOA linkage
- Points of interest from NHS, OSM, and commercial sources
- Air quality surfaces from DEFRA
- NDVI greenspace data via Google Earth Engine

### 2. Routing Cluster Setup (`AHAH_Routing_Cluster_Setup.ipynb`)

Configures a parallel Valhalla routing cluster:

```python
from valhalla_cluster import ValhallaCluster

cluster = ValhallaCluster(
    num_workers=8,              # Parallel routing instances
    base_worker_port=8020,      # Port range start
    tiles_dir="./valhalla_tiles",
    tile_url="https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf"
)

# Build routing tiles (one-time, ~20 minutes)
cluster.build_tiles()
cluster.generate_compose()
cluster.apply_optimisations()
cluster.start()
```

### 3. Accessibility Calculations (`AHAH_Access_Calculations.ipynb`)

Computes travel times from every postcode to nearest POIs:

```python
from poi_accessibility import POIAccessibilityCalculator

calc = POIAccessibilityCalculator(
    postcodes_path="data/postcodes/postcodes.parquet",
    results_dir="data/routing_results/",
    batch_size=50,
    k_nearest=4,
    costing="auto"  # driving mode
)

# Process multiple POI categories
calc.calculate_batch([
    "data/poi/GP.parquet",
    "data/poi/pharmacy.parquet",
    "data/poi/hospital.parquet",
    "data/poi/dentist.parquet",
    "data/poi/leisure.parquet",
    "data/poi/fast_food.parquet",
    "data/poi/gambling.parquet",
    "data/poi/pub_bar.parquet",
    "data/poi/tobacco.parquet",
    "data/poi/bluespace.parquet",
])
```

### 4. Index Construction (`AHAH_Build_Indicator.ipynb`)

Aggregates accessibility measures and constructs the composite index:

```python
from AHAH_Build_Indicator import calculate_ahah

# Aggregate postcode-level results to LSOA means
accessibility = aggregate_routing_results("data/routing_results/")

# Merge with air quality and greenspace
ahah_input = accessibility.merge(air_quality).merge(greenspace)

# Calculate domain scores and composite index
ahah_v5 = calculate_ahah(ahah_input)
ahah_v5.to_csv("AHAH_V5.csv", index=False)
```

## Core Modules

### `valhalla_cluster.py`

Manages a Docker-based Valhalla routing cluster for high-throughput route calculations.

| Method | Description |
|--------|-------------|
| `build_tiles()` | Downloads OSM data and builds routing graph |
| `generate_compose()` | Creates docker-compose.yml for worker cluster |
| `apply_optimisations()` | Tunes Valhalla config for batch processing |
| `start()` / `stop()` | Cluster lifecycle management |
| `health_check()` | Verifies all workers are responsive |

### `poi_accessibility.py`

Calculates travel times from postcodes to points of interest using parallel requests.

| Method | Description |
|--------|-------------|
| `detect_cluster()` | Auto-discovers active Valhalla workers |
| `calculate()` | Process single POI dataset |
| `calculate_batch()` | Process multiple POI datasets with resume support |

## Index Methodology

The AHAH index follows a standardised construction methodology:

1. **Ranking** — Each indicator is ranked across all small areas:
   - Health/access indicators: ascending (lower distance = better)
   - Greenspace (NDVI): descending (higher = better)
   - Hazards: descending (further distance = better)

2. **Normalisation** — Ranks are transformed using the inverse normal (probit) function

3. **Domain Scores** — Transformed indicators are averaged within each domain

4. **Exponential Transformation** — Domain ranks are transformed using:
   ```
   score = -23 × ln(1 - (rank/n) × (1 - e^(-100/23)))
   ```
5. **Composite Score** — Equal-weighted average of four domain scores

Higher AHAH scores indicate less healthy environments.

## Output Variables

The final dataset includes:

| Variable | Description |
|----------|-------------|
| `lsoa21cd` | LSOA/Data Zone code |
| `GP`, `dentist`, etc. | Raw accessibility times (minutes) |
| `*_rnk` | National rank for each indicator |
| `*_pct` | National percentile for each indicator |
| `domain_h`, `domain_g`, `domain_e`, `domain_r` | Domain scores |
| `domain_*_pct` | Domain percentiles |
| `ahah` | Composite AHAH score |
| `ahah_rnk` | National AHAH rank |
| `ahah_pct` | National AHAH percentile (1-100) |

## Data Sources

| Domain | Indicator | Source |
|--------|-----------|--------|
| Health | GP surgeries | NHS Digital |
| Health | Pharmacies | NHS Digital |
| Health | Hospitals | NHS Digital / NHS Scotland |
| Health | Dentists | NHS Digital |
| Health | Leisure centres | OpenStreetMap |
| Green/Blue | Greenspace (Ambient) | Sentinel-2 via Google Earth Engine |
| Green/Blue | Greenspace (Active) | Ordnance Survey Greenspace |
| Green/Blue | Bluespace | OpenStreetMap |
| Environment | NO₂, PM₁₀, SO₂ | DEFRA air quality surfaces |
| Retail | Fast food | OpenStreetMap |
| Retail | Gambling | Gambling Commission |
| Retail | Pubs/bars | OpenStreetMap |
| Retail | Tobacco | Commercial data |

## Citation

If you use this code or the AHAH index, please cite:

> Daras, K., Green, M.A., Davies, A. et al. Open data on health-related neighbourhood features in Great Britain. Sci Data 6, 107 (2019). (Note this is for v1 which used a different methodology.)

## Acknowledgements

- [Valhalla](https://github.com/valhalla/valhalla) — Open source routing engine
- [GIS-OPS Docker Valhalla](https://github.com/gis-ops/docker-valhalla) — Docker image
- [Geofabrik](https://download.geofabrik.de/) — OpenStreetMap extracts
- [Google Earth Engine](https://earthengine.google.com/) — Satellite imagery processing

## Related Resources

- [AHAH Documentation](https://data.geods.ac.uk/dataset/access-to-healthy-assets-hazards-ahah)
