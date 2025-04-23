import pyproj
import streamlit as st
from matplotlib.colors import to_hex, LinearSegmentedColormap
import numpy as np


def latlon_to_xy(lat, lon):
    crs = pyproj.CRS.from_cf(
        {
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": [63.3, 63.3],
            "longitude_of_central_meridian": 15.0,
            "latitude_of_projection_origin": 63.3,
            "earth_radius": 6371000.0,
        }
    )
    # Transformer to project from ESPG:4368 (WGS:84) to our lambert_conformal_conic
    proj = pyproj.Proj.from_crs(4326, crs, always_xy=True)

    # Compute projected coordinates of lat/lon point
    X, Y = proj.transform(lon, lat)
    return X, Y


@st.cache_data(ttl=3600)
def interpolate_color(
    wind_speed,
    thresholds=[2, 4, 5, 14],
    colors=["grey", "green", "orange", "red", "black"],
):
    # Normalize thresholds to range [0, 1]
    norm_thresholds = [t / max(thresholds) for t in thresholds]
    norm_thresholds = [0] + norm_thresholds + [1]

    # Extend color list to match normalized thresholds
    extended_colors = [colors[0]] + colors + [colors[-1]]

    # Create colormap
    cmap = LinearSegmentedColormap.from_list(
        "wind_speed_cmap", list(zip(norm_thresholds, extended_colors)), N=256
    )

    # Normalize wind speed to range [0, 1] and get color
    norm_wind_speed = wind_speed / max(thresholds)
    return to_hex(cmap(np.clip(norm_wind_speed, 0, 1)))
