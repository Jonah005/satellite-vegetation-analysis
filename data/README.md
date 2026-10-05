# Dataset

The original Sentinel-2 GeoTIFF imagery used in this project is not included in the repository because the raster files are large.

The analysis expects NDVI GeoTIFF files for the two comparison years:

```text
2018/
└── <2018 NDVI GeoTIFF>

2024/
└── <2024 NDVI GeoTIFF>
```

The processing script searches for filenames containing `NDVI` and supports both `.tif` and `.tiff` files.

## Data Used

The project compares Sentinel-2 NDVI imagery from:

- **2018**
- **2024**

The 2024 raster is aligned to the 2018 raster grid when the coordinate reference system, transform, width or height differ.

## NDVI Processing

The analysis uses the original NDVI values for statistics and change calculations.

The vegetation-change raster is calculated as:

```text
NDVI change = NDVI 2024 - NDVI 2018
```

The project then calculates global statistics, regional 3×3 grid values, vegetation gain/loss counts and spatial hotspots.

## Adding Your Own Data

To reproduce the analysis, place the corresponding NDVI GeoTIFF files in the year folders expected by the script and keep the raw raster files outside Git if they are large.

Recommended `.gitignore` rules:

```gitignore
*.tif
*.tiff
```
