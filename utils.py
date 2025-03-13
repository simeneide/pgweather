import pyproj


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
