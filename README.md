# Satellite Vegetation Analysis with LLM-Assisted Interpretation

A satellite-based vegetation change analysis project that compares **Sentinel-2 NDVI imagery from 2018 and 2024** and uses an LLM-assisted interpretation layer to summarize the measured changes.

The system aligns the NDVI rasters, computes vegetation change directly from the original NDVI arrays, identifies gain and loss regions, generates comparison visualizations, and produces a constrained natural-language summary based on the calculated statistics.

## What the Project Does

- Loads Sentinel-2 NDVI GeoTIFF imagery for 2018 and 2024.
- Aligns the newer raster to the 2018 reference grid when required.
- Computes NDVI change as `2024 NDVI - 2018 NDVI`.
- Calculates overall NDVI statistics and change distributions.
- Splits the image into a 3×3 spatial grid for regional comparison.
- Detects the strongest vegetation-gain and vegetation-loss hotspots.
- Generates side-by-side NDVI and NDVI-difference visualizations.
- Uses a local LLM/vision model to interpret the calculated results.
- Validates the model output against the measured NDVI values.
- Falls back to a deterministic statistics-based report if the model output is invalid.

## Project Structure

```text
satellite-vegetation-analysis/
├── src/
│   └── compare_ndvi_years.py
├── outputs/
│   ├── facts_2018_2024.json
│   ├── ndvi_2018_2024_side_by_side.png
│   ├── ndvi_diff_2018_2024.png
│   └── summary_2018_2024.md
├── data/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Analysis Pipeline

```text
Sentinel-2 NDVI GeoTIFFs
        ↓
Raster Loading
        ↓
Spatial Alignment
        ↓
NDVI Difference Calculation
        ↓
Global Statistics + 3×3 Grid Analysis
        ↓
Gain/Loss Hotspot Detection
        ↓
Visualisation Generation
        ↓
LLM-Assisted Interpretation
        ↓
Output Validation
        ↓
Final Summary
```

## Outputs

The project produces:

- `facts_2018_2024.json` — calculated NDVI statistics, grid values, thresholds and hotspot information.
- `ndvi_2018_2024_side_by_side.png` — visual comparison of the 2018 and 2024 NDVI rasters using a shared scale.
- `ndvi_diff_2018_2024.png` — vegetation-change map showing the NDVI difference between the two years.
- `summary_2018_2024.md` — final validated interpretation of the measured vegetation changes.

## LLM-Assisted Interpretation

The interpretation layer uses a local model through Ollama.

Instead of allowing the model to freely describe the satellite imagery, the model is given the calculated NDVI statistics and is instructed to base its explanation only on those values.

The generated response is then checked against the actual mean NDVI values. If the response is inconsistent or contains disallowed visual guesses, the system uses a deterministic statistics-based summary instead.

## Technologies

- Python
- NumPy
- Rasterio
- Matplotlib
- Pillow
- Ollama
- Moondream
- Sentinel-2 imagery
- NDVI
- Geospatial raster analysis

## Data

The original Sentinel-2 GeoTIFF files are not included in this repository because of their size.

See [`data/README.md`](data/README.md) for details.
