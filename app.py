# %%
import xarray as xr
from siphon.catalog import TDSCatalog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import streamlit as st
import datetime
from scipy.interpolate import griddata
import branca.colormap as cm
import os
from utils import latlon_to_xy
import plotly.graph_objects as go
from matplotlib.colors import to_hex, LinearSegmentedColormap
from plotly.subplots import make_subplots


@st.cache_data(ttl=7200)
def load_data():
    """
    Loads a NetCDF file containing forecast data. Example
    <xarray.Dataset> Size: 483MB
    Dimensions:               (altitude: 34, time: 67, y: 81, x: 81)
    Coordinates:
        height                float32 4B 0.0
        hybrid                (altitude) float64 272B 0.6784 0.6984 ... 0.9985
    * x                     (x) float32 324B -5.101e+05 -5.076e+05 ... -3.101e+05
    * y                     (y) float32 324B -2.825e+05 -2.8e+05 ... -8.252e+04
    * time                  (time) datetime64[ns] 536B 2025-03-13T12:00:00 ... ...
        longitude             (y, x) float64 52kB 5.684 5.729 5.774 ... 8.919 8.967
        latitude              (y, x) float64 52kB 60.43 60.43 60.43 ... 62.42 62.43
    * altitude              (altitude) float64 272B 2.89e+03 2.684e+03 ... 11.66
    Data variables:
        ap                    (altitude) float64 272B 5.682e+03 5.01e+03 ... 0.0 0.0
        b                     (altitude) float64 272B 0.6223 0.6489 ... 0.9985
        air_temperature_ml    (time, altitude, y, x) float32 60MB 252.5 ... 260.7
        x_wind_ml             (time, altitude, y, x) float32 60MB -1.6 ... 5.709
        y_wind_ml             (time, altitude, y, x) float32 60MB -11.19 ... -10.67
        surface_air_pressure  (time, y, x, altitude) float32 60MB 9.46e+04 ... 8....
        elevation             (y, x, altitude) float32 892kB 465.8 ... 1.543e+03
        air_temperature_0m    (time, y, x, altitude) float32 60MB 278.6 ... 261.1
        wind_speed            (time, altitude, y, x) float32 60MB 11.31 ... 12.1
        thermal_temp_diff     (time, y, x, altitude) float64 120MB 2.331 ... 0.3471
        thermal_top           (time, y, x) float64 4MB 2.89e+03 ... 2.89e+03
    Attributes: (12/41)
        min_time:                    2025-03-13T12:00:00Z
        geospatial_lat_min:          49.8
        geospatial_lat_max:          75.2
        geospatial_lon_min:          -18.1
        geospatial_lon_max:          54.2
        comment:                     For more information, please visit https://g...
        ...                          ...
        publisher_name:              Norwegian Meteorological Institute
        summary:                     This file contains model level parameters fr...
        summary_no:                  Denne filen inneholder modelnivåparametere f...
        title:                       Meps 2.5Km deterministic model level paramet...
        title_no:                    Meps 2.5Km deterministisk modellnivåparamete...
        related_dataset:             no.met:8c94c7de-6328-4113-9e77-8f090999fab9 ...
    """
    # Find all files in the forecasts directory
    forecast_dir = "forecasts"
    # Get a list of all NetCDF files
    nc_files = [f for f in os.listdir(forecast_dir) if f.endswith(".nc")]
    if not nc_files:
        raise FileNotFoundError("No forecast files found in the 'forecasts' directory")

    # Sort files by their timestamp, assuming the filenames contain the timestamp
    nc_files.sort()

    # Choose the latest file
    latest_file = os.path.join(forecast_dir, nc_files[-1])

    # Load the dataset from the latest file
    subset = xr.open_dataset(latest_file)
    return subset


def wind_and_temp_colorscales(wind_max=20, tempdiff_max=8):
    # build colorscale for thermal temperature difference
    wind_colors = ["grey", "blue", "green", "yellow", "red", "purple"]
    wind_positions = [0, 0.5, 3, 7, 12, 20]  # transition points
    wind_positions_norm = [i / wind_max for i in wind_positions]

    # Create the colormap
    windcolors = mcolors.LinearSegmentedColormap.from_list(
        "", list(zip(wind_positions_norm, wind_colors))
    )

    # build colorscale for thermal temperature difference
    thermal_colors = ["white", "white", "red", "violet", "darkviolet"]
    thermal_positions = [0, 0.2, 2.0, 4, 8]
    thermal_positions_norm = [i / tempdiff_max for i in thermal_positions]

    # Create the colormap
    tempcolors = mcolors.LinearSegmentedColormap.from_list(
        "", list(zip(thermal_positions_norm, thermal_colors))
    )
    return windcolors, tempcolors


@st.cache_data(ttl=60)
def create_wind_map(
    _subset, x_target, y_target, altitude_max=4000, date_start=None, date_end=None
):
    """
    _subset = subset
    altitude_max = 3000
    x_target = -422175.14005226345
    y_target = -204279.84596708667


    """
    subset = _subset

    wind_min, wind_max = 0.3, 20
    tempdiff_min, tempdiff_max = 0, 8
    wind_colors = ["grey", "blue", "green", "yellow", "red", "purple"]

    if date_start is None:
        date_start = datetime.datetime.fromtimestamp(
            subset.time.min().values.astype("int64") // 1e9
        )
    if date_end is None:
        date_end = datetime.datetime.fromtimestamp(
            subset.time.max().values.astype("int64") // 1e9
        )

    # Resample time and altitude for the wind plot data.
    new_timestamps = pd.date_range(date_start, date_end, 20)
    new_altitude = np.arange(subset.elevation.mean(), altitude_max, altitude_max / 20)

    windplot_data = subset.sel(x=x_target, y=y_target, method="nearest")
    windplot_data = windplot_data.interp(altitude=new_altitude, time=new_timestamps)

    # Convert data for Plotly heatmap
    thermal_diff = windplot_data["thermal_temp_diff"].T.values
    times = [pd.Timestamp(time).strftime("%H:%M") for time in windplot_data.time.values]
    altitudes = windplot_data.altitude.values

    # Creating Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=thermal_diff,
            x=times,
            y=altitudes,
            colorscale="YlGn",
            colorbar=dict(title="Thermal Temperature Difference (°C)"),
            zmin=tempdiff_min,
            zmax=tempdiff_max,
        )
    )

    # Add wind quiver plots (Note: Plotly doesn't support quivers directly like matplotlib; consider using streamlines or other visualization methods for precise vector representation).
    speed = np.sqrt(windplot_data["x_wind_ml"] ** 2 + windplot_data["y_wind_ml"] ** 2).T
    fig.add_trace(
        go.Scatter(
            x=times,
            y=altitudes,
            mode="markers",
            marker=dict(
                size=8,
                color=speed,
                colorscale=wind_colors,
                colorbar=dict(title="Wind Speed (m/s)"),
            ),
            # text=[f"Speed: {s:.2f} m/s" for s in speed.squeeze()],
            hoverinfo="text",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Wind and Thermals Starting at {date_start.strftime('%Y-%m-%d')} (UTC)",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Altitude (m)"),
    )

    return fig


# %%
@st.cache_data(ttl=7200)
def create_sounding(_subset, date, hour, x_target, y_target, altitude_max=3000):
    """
    date = "2024-05-12"
    hour = "15"
    x_target = 5
    y_target = 5
    """
    subset = _subset
    lapse_rate = 0.0098  # in degrees Celsius per meter
    subset = subset.where(subset.altitude < altitude_max, drop=True)
    # Create a figure object
    fig, ax = plt.subplots()

    # Define the dry adiabatic lapse rate
    def add_dry_adiabatic_lines(ds):
        # Define a range of temperatures at sea level
        T0 = np.arange(-40, 40, 5)  # temperatures from -40°C to 40°C in steps of 10°C

        # Create a 2D grid of temperatures and altitudes
        T0, altitude = np.meshgrid(T0, ds.altitude)

        # Calculate the temperatures at each altitude
        T_adiabatic = T0 - lapse_rate * altitude

        # Plot the dry adiabatic lines
        for i in range(T0.shape[1]):
            ax.plot(T_adiabatic[:, i], ds.altitude, "r:", alpha=0.5)

    # Plot the actual temperature profiles
    time_str = f"{date} {hour}:00:00"
    # find x and y values cloeset to given latitude and longitude

    ds_time = subset.sel(time=time_str, x=x_target, y=y_target, method="nearest")
    T = ds_time["air_temperature_ml"].values - 273.3  # in degrees Celsius
    ax.plot(
        T, ds_time.altitude, label=f"temp {pd.to_datetime(time_str).strftime('%H:%M')}"
    )

    # Define the surface temperature
    T_surface = T[-1] + 3
    T_parcel = T_surface - lapse_rate * ds_time.altitude

    # Plot the temperature of the rising air parcel
    filter = T_parcel > T
    ax.plot(
        T_parcel[filter],
        ds_time.altitude[filter],
        label="Rising air parcel",
        color="green",
    )

    add_dry_adiabatic_lines(ds_time)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        f"Temperature Profile and Dry Adiabatic Lapse Rate for {date} {hour}:00"
    )
    ax.legend(title="Time")
    xmin, xmax = (
        ds_time["air_temperature_ml"].min().values - 273.3,
        ds_time["air_temperature_ml"].max().values - 273.3 + 3,
    )
    ax.set_xlim(xmin, xmax)
    ax.grid(True)

    # Return the figure object
    return fig


# %%
def date_controls(subset):
    start_stop_time = [
        subset.time.min().values.astype("M8[D]").astype("O"),
        subset.time.max().values.astype("M8[D]").astype("O"),
    ]
    now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0).date()

    if "forecast_date" not in st.session_state:
        st.session_state.forecast_date = now
    if "forecast_time" not in st.session_state:
        st.session_state.forecast_time = datetime.time(14, 0)
    if "altitude_max" not in st.session_state:
        st.session_state.altitude_max = 3000
    if "target_latitude" not in st.session_state:
        st.session_state.target_latitude = 61.22908
    if "target_longitude" not in st.session_state:
        st.session_state.target_longitude = 7.09674

    # Generate available days within the dataset's time range
    available_days = pd.date_range(
        start=start_stop_time[0], end=start_stop_time[1]
    ).date

    day_cols = st.columns(len(available_days))  # Create columns for each available day

    for i, day in enumerate(available_days):
        label = day.strftime("%A")  # Get day label
        if day == now:
            label += " (today)"
        with day_cols[i]:  # Place each button in its respective column
            if st.button(label):
                st.session_state.forecast_date = day

    # Group hours into smaller rows for better layout
    hours_per_row = 24  # Define how many hour buttons to display per row
    available_hours = range(7, 22, 1)  # 24-hour format

    # Divide hours into batches
    hour_batches = [
        available_hours[i : i + hours_per_row]
        for i in range(0, len(available_hours), hours_per_row)
    ]

    # Display hour buttons in rows
    for batch in hour_batches:
        hour_cols = st.columns(len(batch))
        for i, hour in enumerate(batch):
            label = f"{hour:02}:00"
            with hour_cols[i]:
                if st.button(label):
                    st.session_state.forecast_time = datetime.time(hour, 0)


def build_map(_subset, date=None, hour=None):
    subset = _subset

    latitude_values = subset.latitude.values.flatten()
    longitude_values = subset.longitude.values.flatten()
    thermal_top_values = (
        subset.thermal_top.sel(time=f"{date}T{hour}").values.flatten().round()
    )

    # Use Plotly's scattermap for visualization and enable click events
    scatter_map = go.Scattermap(
        lat=latitude_values,
        lon=longitude_values,
        mode="markers",
        marker=go.scattermap.Marker(
            size=9,
            color=thermal_top_values,
            colorscale="Viridis",
            colorbar=dict(title="Thermal Height (m)"),
        ),
        text=[f"Thermal Height: {ht} m" for ht in thermal_top_values],
        hoverinfo="text",
    )

    fig = go.Figure(scatter_map)

    fig.update_layout(
        map_style="open-street-map",
        map=dict(center=dict(lat=61.22908, lon=7.09674), zoom=9),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    # Register click event callback

    return fig


def interpolate_color(
    wind_speed, thresholds=[2, 8, 14], colors=["white", "green", "red", "black"]
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


def create_daily_thermal_and_wind_airgram(subset, x_target, y_target, date):
    """
    Create a Plotly subplot figure for a single day's thermal and wind data.
    The top subplot shows wind data as arrows for direction and color for strength.
    The bottom subplot shows thermal temperature differences.
    """
    # Define the time window to display
    display_start_hour = 7
    display_end_hour = 21

    # Extract the day that matches the provided date
    start_date = pd.Timestamp(date).normalize()
    end_date = start_date + pd.Timedelta(days=1)

    # Select data for the given date
    daily_data = subset.sel(time=slice(start_date, end_date))

    # Create time mask for the given display window
    time_values = pd.to_datetime(
        daily_data.time.values
    )  # Convert numpy.datetime64 to datetime

    mask = [(display_start_hour <= t.hour < display_end_hour) for t in time_values]

    # Filter data within the specified hours
    daily_data = daily_data.isel(time=mask)

    # Select nearest points for the supplied x and y indices
    location_data = daily_data.sel(x=x_target, y=y_target, method="nearest")

    # Interpolating the data for visualization
    new_timestamps = pd.date_range(
        start=start_date, end=end_date, freq="h"
    )  # Every full hour

    # Remove timestamps that are outside the range of the data
    new_timestamps = new_timestamps[
        (new_timestamps >= location_data.time.min().values)
        & (new_timestamps <= location_data.time.max().values)
    ]

    altitudes_thermal = np.arange(0, 3000, 200)  # Every 200 meters
    altitudes_thermal = altitudes_thermal[
        (altitudes_thermal >= location_data.altitude.min().values)
        & (altitudes_thermal <= location_data.altitude.max().values)
    ]

    # Interpolate thermal temperature difference for the specified times and altitudes
    thermal_diff = (
        location_data["thermal_temp_diff"]
        .interp(time=new_timestamps, altitude=altitudes_thermal)
        .T.values
    )

    # Generating time labels for the x-axis
    times = [t.strftime("%H:%M") for t in new_timestamps]

    # Calculate wind data at 500m intervals
    altitudes_wind = np.arange(0, 3000, 500)
    altitudes_wind = altitudes_wind[
        (altitudes_wind >= location_data.altitude.min().values)
        & (altitudes_wind <= location_data.altitude.max().values)
    ]

    x_wind = location_data["x_wind_ml"].interp(
        time=new_timestamps, altitude=altitudes_wind
    )
    y_wind = location_data["y_wind_ml"].interp(
        time=new_timestamps, altitude=altitudes_wind
    )

    # Calculate wind speed and direction
    speed = np.sqrt(x_wind**2 + y_wind**2).T.values
    angles = np.rad2deg(np.arctan2(y_wind, x_wind)).T.values  # Convert to degrees
    angles = angles = (angles + 90) % 360
    # Create a subplot figure with shared x-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.05,
        subplot_titles=("Wind Speed and Direction", "Thermal Temperature Difference"),
    )

    # Add wind data plot as rotated triangular markers with a common legend
    for i, alt in enumerate(altitudes_wind):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[alt] * len(times),
                mode="markers",
                marker=dict(
                    symbol="arrow",
                    size=20,
                    angle=angles[i],
                    color=[interpolate_color(s) for s in speed[i]],
                    # colorscale="Viridis",
                    showscale=False,  # Hide individual color scales
                    cmin=0,
                    cmax=20,
                ),
                hoverinfo="text",
                text=[
                    f"Alt: {alt} m, Speed: {spd:.1f} m/s, Direction: {angle:.1f}°"
                    for spd, angle in zip(speed[i], angles[i])
                ],
            ),
            row=1,
            col=1,
        )
        fig.update_layout(showlegend=False)

    # Add a legend indicator for the wind speed at the right of the plots
    fig.add_shape(
        type="rect",
        x0=1.05,
        y0=0.2,
        x1=1.10,
        y1=0.8,
        xref="paper",
        yref="paper",
        line=dict(width=0),
        fillcolor="rgba(0,0,0,0)",
    )

    annotations = [
        dict(
            x=1.15,
            y=y,
            text=f"{int(s)} m/s",
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        for y, s in zip(np.linspace(0.2, 0.8, 5), range(0, 20, 5))
    ]
    fig.update_layout(annotations=annotations)

    # Add thermal data plot
    fig.add_trace(
        go.Heatmap(
            z=thermal_diff,
            x=times,
            y=altitudes_thermal,
            colorscale="YlGn",
            colorbar=dict(
                title="Thermal Temp Difference (°C)",
                thickness=10,
                ypad=75,  # Moves the color bar vertically
            ),
            zmin=0,
            zmax=8,
            text=thermal_diff.round(1),
            texttemplate="%{text}",
            textfont={"size": 12},
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=950,
        title=f"Airgram for {start_date.strftime('%Y-%m-%d')}, lat/lon: {st.session_state.target_latitude:.2f}, {st.session_state.target_longitude:.2f}",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Altitude (m)"),
        xaxis2=dict(title="Time", tickangle=-45),
        yaxis1=dict(title="Altitude (m)", range=[0, 3000]),
    )

    return fig


def create_daily_airgram(subset, x_target, y_target, date):
    """
    Create a Plotly heatmap for a single day's wind and thermal data.

    :param subset: xarray Dataset containing the weather data.
    :param x_target: The x-coordinate index (not longitude) of the target location.
    :param y_target: The y-coordinate index (not latitude) of the target location.
    :param date: The specific date for which the data is visualized (datetime object).
    :return: A Plotly figure object.
    """
    # Define the time window to display
    display_start_hour = 7
    display_end_hour = 21

    # Extract the day that matches the provided date
    start_date = pd.Timestamp(date).normalize()
    end_date = start_date + pd.Timedelta(days=1)

    # Select data for the given date
    daily_data = subset.sel(time=slice(start_date, end_date))

    # Create time mask for the given display window
    time_values = pd.to_datetime(
        daily_data.time.values
    )  # Convert numpy.datetime64 to datetime
    mask = [(display_start_hour <= t.hour < display_end_hour) for t in time_values]

    # Filter data within the specified hours
    daily_data = daily_data.isel(time=mask)
    # Select nearest points for the supplied x and y indices
    location_data = daily_data.sel(x=x_target, y=y_target, method="nearest")

    # Interpolating the data for visualization
    new_timestamps = pd.date_range(
        start=start_date, end=end_date, freq="h"
    )  # Every full hour

    # Remove timestamps that are outside the range of the data
    new_timestamps = new_timestamps[
        (new_timestamps >= location_data.time.min().values)
        & (new_timestamps <= location_data.time.max().values)
    ]

    altitudes = np.arange(0, 3000, 200)  # Every 200 meters
    # Remove altitude that are outside the range of the data
    altitudes = altitudes[
        (altitudes >= location_data.altitude.min().values)
        & (altitudes <= location_data.altitude.max().values)
    ]

    # Interpolate thermal temperature difference for the specified times and altitudes
    thermal_diff = (
        location_data["thermal_temp_diff"]
        .interp(time=new_timestamps, altitude=altitudes)
        .T.values
    )

    # Generating time labels for the x-axis
    times = [t.strftime("%H:%M") for t in new_timestamps]

    # Creating Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=thermal_diff,
            x=times,
            y=altitudes,
            colorscale="YlGn",
            colorbar=dict(title="Thermal Temperature Difference (°C)"),
            zmin=0,
            zmax=8,  # Adjusted for expected data range
            text=thermal_diff.round(1),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    # Update layout
    fig.update_layout(
        title=f"Thermal Profiles for {start_date.strftime('%Y-%m-%d')}",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Altitude (m)"),
        xaxis_tickangle=-45,
    )
    return fig


def show_forecast():
    subset = load_data()

    date_controls(subset)
    # time_start = datetime.time(0, 0)
    # # convert subset.attrs['min_time']='2024-05-11T06:00:00Z' into datetime
    # min_time = datetime.datetime.strptime(
    #     subset.attrs["min_time"], "%Y-%m-%dT%H:%M:%SZ"
    # )
    # date_start = datetime.datetime.combine(st.session_state.forecast_date, time_start)
    # date_start = max(date_start, min_time)

    ## MAP
    with st.expander("Map", expanded=True):
        map_fig = build_map(
            _subset=subset,
            date=st.session_state.forecast_date,
            hour=st.session_state.forecast_time,
        )
        map_selection = st.plotly_chart(
            map_fig,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": False},
            on_select="rerun",
        )
        # Update lat lon if selection is made
        selected_points = map_selection.get("selection").get("points")
        if len(selected_points) > 0:
            point = selected_points[0]
            st.session_state.target_latitude = point["lat"]
            st.session_state.target_longitude = point["lon"]
            print("Updated lat lon")
    x_target, y_target = latlon_to_xy(
        st.session_state.target_latitude, st.session_state.target_longitude
    )
    wind_fig = create_daily_thermal_and_wind_airgram(
        subset,
        x_target=x_target,
        y_target=y_target,
        date=st.session_state.forecast_date,
    )
    # wind_fig = create_wind_map(
    #     subset,
    #     date_start=date_start,
    #     date_end=date_end,
    #     altitude_max=st.session_state.altitude_max,
    #     x_target=x_target,
    #     y_target=y_target,
    # )
    st.plotly_chart(wind_fig)
    plt.close()

    with st.expander("More settings", expanded=False):
        st.session_state.altitude_max = st.number_input(
            "Max altitude", 0, 4000, 3000, step=500
        )

    ############################
    ######### SOUNDING #########
    ############################
    st.markdown("---")
    with st.expander("Sounding", expanded=False):
        date = datetime.datetime.combine(
            st.session_state.forecast_date, st.session_state.forecast_time
        )

        with st.spinner("Building sounding..."):
            sounding_fig = create_sounding(
                subset,
                date=date.date(),
                hour=date.hour,
                altitude_max=st.session_state.altitude_max,
                x_target=x_target,
                y_target=y_target,
            )
        st.pyplot(sounding_fig)
        plt.close()

    st.markdown(
        "Wind and sounding data from MEPS model (main model used by met.no), including the estimated ground temperature. Ive probably made many errors in this process."
    )


if __name__ == "__main__":
    run_streamlit = True
    if run_streamlit:
        st.set_page_config(page_title="PGWeather", page_icon="🪂", layout="wide")
        show_forecast()
    else:
        lat = 61.22908
        lon = 7.09674
        x_target, y_target = latlon_to_xy(lat, lon)

        build_map_overlays(subset, date="2024-05-14", hour="16")

        wind_fig = create_wind_map(
            subset, altitude_max=3000, x_target=x_target, y_target=y_target
        )

        # Plot thermal top on a map for a specific time
        # subset.sel(time=subset.time.min()).thermal_top.plot()
        sounding_fig = create_sounding(
            subset, date="2024-05-12", hour=15, x_target=x_target, y_target=y_target
        )
