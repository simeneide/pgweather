import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import streamlit as st
import datetime
import os
from utils import latlon_to_xy
import plotly.graph_objects as go
from matplotlib.colors import to_hex, LinearSegmentedColormap
from plotly.subplots import make_subplots
import db_utils
import polars as pl


def initialize_query_params(df_forecast):
    # Initialize latitude and longitude from query params or set defaults
    if "lat" in st.query_params and "lon" in st.query_params:
        st.session_state.target_latitude = float(st.query_params["lat"])
        st.session_state.target_longitude = float(st.query_params["lon"])
    else:
        # Default value if not present in the query params
        st.session_state.target_latitude = None
        st.session_state.target_longitude = None


# Function to update query params
def update_query_params(lat, lon):
    st.query_params.lat = str(lat)
    st.query_params.lon = str(lon)


@st.cache_data(ttl=7200)
def load_data():
    """
    Connects to the database and loads the forecast data as a Polars DataFrame.

    # For dev:
    # Save to local file
    df_forecast.write_parquet("tmp_weather_forecasts.parquet")

    # load from local file
    df_forecast = pl.read_parquet("tmp_weather_forecasts.parquet")
    """
    with st.spinner("Loading forecast from database...", show_time=True):
        db = db_utils.Database()

        # Read the data from the database
        query = """
        SELECT forecast_timestamp, time, altitude, air_temperature_ml, x_wind_ml,
            y_wind_ml, longitude, latitude, wind_speed, thermal_temp_diff, thermal_top
        FROM weather_forecasts
        """

        df_forecast = db.read(query)  # Using the Polars DataFrame
        df_forecast = df_forecast.with_columns(
            [
                pl.col("forecast_timestamp").cast(pl.Datetime),
                pl.col("time").cast(pl.Datetime),
            ]
        )

    return df_forecast


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
    df_forecast, x_target, y_target, altitude_max=4000, date_start=None, date_end=None
):
    """
    Generates a heatmap of wind and thermals.
    """
    wind_min, wind_max = 0.3, 20
    tempdiff_min, tempdiff_max = 0, 8
    wind_colors = ["grey", "blue", "green", "yellow", "red", "purple"]

    if date_start is None:
        date_start = df_forecast.get_column("time").min().to_pandas()
    if date_end is None:
        date_end = df_forecast.get_column("time").max().to_pandas()

    # Create the data subset
    subset = df_forecast.filter(
        (pl.col("longitude") == x_target) & (pl.col("latitude") == y_target)
    )

    # Since Polars doesn't directly work with datetime ranges like xarray, use pandas
    new_timestamps = pd.date_range(date_start, date_end, 20)
    new_altitude = np.linspace(
        subset.get_column("altitude").mean(), altitude_max, num=20
    )

    # Simulate resampling, normally not directly possible in Polars
    windplot_data = subset.to_pandas()
    windplot_data["time"] = pd.to_datetime(windplot_data["time"])
    windplot_data = windplot_data.set_index(["time", "altitude"]).sort_index()

    windplot_data = windplot_data.interp(time=new_timestamps, method="linear").interp(
        altitude=new_altitude, method="linear"
    )

    # Convert data for Plotly heatmap
    thermal_diff = windplot_data["thermal_temp_diff"].T.values
    times = [pd.Timestamp(t).strftime("%H:%M") for t in new_timestamps]
    altitudes = new_altitude

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

    # Add wind quiver plots
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


@st.cache_data(ttl=7200)
def create_sounding(_subset, date, hour, lon, lat, altitude_max=3000):
    """
    Create a sounding plot using the Polars DataFrame.
    """
    lapse_rate = 0.0098  # in degrees Celsius per meter
    subset = _subset.filter(pl.col("altitude") < altitude_max)
    # Create a figure object
    fig, ax = plt.subplots()

    # Define the dry adiabatic lapse rate
    hour = datetime.time(hour, 0, 0)
    map_datetime = datetime.datetime.combine(date, hour)

    df_time = subset.filter(
        (pl.col("time") == map_datetime)
        & (pl.col("longitude") == lon)
        & (pl.col("latitude") == lat)
    ).sort("altitude")

    time_data = df_time.to_pandas()
    altitudes = time_data["altitude"].values
    temperatures = time_data["air_temperature_ml"].values - 273.3

    def add_dry_adiabatic_lines(ax, alts, lapse_rate):
        T0 = np.arange(-40, 40, 5)
        T0, altitude = np.meshgrid(T0, alts)
        T_adiabatic = T0 - lapse_rate * altitude

        for i in range(T0.shape[1]):
            ax.plot(T_adiabatic[:, i], alts, "r:", alpha=0.5)

    ax.plot(
        temperatures, altitudes, label=f"temp {pd.to_datetime(date).strftime('%H:%M')}"
    )

    # Plot the temperature of the rising air parcel
    T_surface = temperatures[-1] + 3
    T_parcel = T_surface - lapse_rate * altitudes
    filter = T_parcel > temperatures
    ax.plot(
        T_parcel[filter],
        altitudes[filter],
        label="Rising air parcel",
        color="green",
    )

    add_dry_adiabatic_lines(ax, altitudes, lapse_rate)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        f"Temperature Profile and Dry Adiabatic Lapse Rate for {date} {hour}:00"
    )
    ax.legend(title="Time")
    xmin, xmax = temperatures.min() - 3, temperatures.max() + 3
    ax.set_xlim(xmin, xmax)
    ax.grid(True)

    return fig


def date_controls(df_forecast):
    start_stop_time = [
        df_forecast.get_column("time").min().date(),
        df_forecast.get_column("time").max().date(),
    ]
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    tomorrow_date = tomorrow.date()

    if "forecast_date" not in st.session_state:
        st.session_state.forecast_date = tomorrow_date
    if "forecast_time" not in st.session_state:
        st.session_state.forecast_time = datetime.time(14, 0)
    if "altitude_max" not in st.session_state:
        st.session_state.altitude_max = 3000
    if "target_latitude" not in st.session_state:
        st.session_state.target_latitude = (
            df_forecast.select("latitude").median().item()
        )
    if "target_longitude" not in st.session_state:
        st.session_state.target_longitude = (
            df_forecast.select("longitude").median().item()
        )

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
            if st.button(
                label,
                type="primary"
                if day == st.session_state.forecast_date
                else "secondary",
            ):
                st.session_state.forecast_date = day
                st.rerun()

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
                if st.button(
                    label,
                    type="primary"
                    if hour == st.session_state.forecast_time.hour
                    else "secondary",
                ):
                    st.session_state.forecast_time = datetime.time(hour, 0)
                    st.rerun()


def build_map(df_forecast, selected_lat=None, selected_lon=None, date=None, hour=None):
    """
    date = datetime.datetime.now().replace(minute=0, second=0, microsecond=0).date()
    hour = datetime.datetime.now().replace(minute=0, second=0, microsecond=0).hour
    hour = datetime.time(hour, 0, 0)
    selected_lon = df_forecast.get_column("longitude").to_numpy()[0]
    selected_lat = df_forecast.get_column("latitude").to_numpy()[0]
    """
    map_datetime = datetime.datetime.combine(date, hour)
    subset = df_forecast.filter((pl.col("time") == map_datetime))

    latitude_values = subset.get_column("latitude").to_numpy()
    longitude_values = subset.get_column("longitude").to_numpy()
    thermal_top_values = subset.get_column("thermal_top").to_numpy().round()

    # Determine whether a point is selected
    selected_points = (latitude_values == selected_lat) & (
        longitude_values == selected_lon
    )

    # Use conditional logic to define marker properties
    marker_size = np.where(selected_points, 20, 9)  # Larger size for selected point

    scatter_map = go.Scattermap(
        lat=latitude_values,
        lon=longitude_values,
        mode="markers",
        marker=go.scattermap.Marker(
            size=marker_size,
            color=thermal_top_values,
            colorscale="Viridis",
            # showscale=False,
            colorbar=dict(title="Thermal Height (m)"),
        ),
        text=[f"Thermal Height: {ht} m" for ht in thermal_top_values],
        hoverinfo="text",
    )

    fig = go.Figure(scatter_map)
    fig.update_layout(
        map_style="open-street-map",
        map=dict(center=dict(lat=selected_lat, lon=selected_lon), zoom=8),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

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


def create_daily_thermal_and_wind_airgram(df_forecast, lat, lon, date):
    """
    Create a Plotly subplot figure for a single day's thermal and wind data.
    The top subplot shows wind data as arrows for direction and color for strength.
    The bottom subplot shows thermal temperature differences.

    lat = df_forecast["latitude"].median()
    lon = df_forecast["longitude"].median()
    date = df_forecast["time"].median().date()

    """
    display_start_hour = 7
    display_end_hour = 21
    prec = 1e-2  # location precision
    location_data = (
        df_forecast.filter(
            # Ensure correct date
            (pl.col("time").dt.date() == date)
            # Ensure correct hours
            & (
                pl.col("time")
                .dt.hour()
                .is_between(display_start_hour, display_end_hour)
            )
        )
        # Ensure correct location
        .filter(
            (pl.col("longitude").is_between(lon - prec, lon + prec))
            & (pl.col("latitude").is_between(lat - prec, lat + prec))
        )
    )

    # Create an empty polar frame with houry spaced timestamps and altitude spaced altitudes
    # Interpolate the data to this frame
    new_timestamps = location_data.select("time").to_series().unique().to_list()
    altitudes = np.arange(0.0, 3000.0, 200)
    # remove altitudes outside of the data
    altitudes = altitudes[altitudes >= location_data["altitude"].min()]

    output_frame = (
        pl.DataFrame(
            {
                "time": [new_timestamps],
                "altitude": [altitudes],
                "lat": lat,
                "lon": lon,
            }
        )
        .explode("time")
        .explode("altitude")
        .sort("altitude")
    )

    # Interpolate the data to this frame

    plot_frame = (
        output_frame.join_asof(
            location_data.sort("altitude"), on="altitude", by="time", strategy="nearest"
        )
        .with_columns(wind_direction=pl.arctan2("y_wind_ml", "x_wind_ml").degrees())
        .sort("time")
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.05,
        subplot_titles=("Wind Speed and Direction", "Thermal Temperature Difference"),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_frame.select("time").to_numpy().squeeze(),
            y=plot_frame.select("altitude").to_numpy().squeeze(),
            mode="markers",
            marker=dict(
                symbol="arrow",
                size=20,
                angle=plot_frame.select("wind_direction").to_numpy().squeeze(),
                color=[
                    interpolate_color(s)
                    for s in plot_frame.select("wind_speed").to_numpy().squeeze()
                ],
                showscale=False,
                cmin=0,
                cmax=20,
            ),
            hoverinfo="text",
            text=[
                f"Alt: {alt} m, Speed: {spd:.1f} m/s, Direction: {angle:.1f}°"
                for alt, spd, angle in zip(
                    plot_frame.select("altitude").to_numpy().squeeze(),
                    plot_frame.select("wind_speed").to_numpy().squeeze(),
                    plot_frame.select("wind_direction").to_numpy().squeeze(),
                )
            ],
        ),
        row=1,
        col=1,
    )
    fig.update_layout(showlegend=False)

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

    fig.add_trace(
        go.Heatmap(
            z=plot_frame.select("thermal_temp_diff").to_numpy().squeeze(),
            x=plot_frame.select("time").to_numpy().squeeze(),
            y=plot_frame.select("altitude").to_numpy().squeeze(),
            colorscale="YlGn",
            showscale=False,
            colorbar=dict(
                title="Thermal Temp Difference (°C)",
                thickness=10,
                ypad=75,
            ),
            zmin=0,
            zmax=8,
            text=plot_frame.select("thermal_temp_diff").to_numpy().squeeze().round(1),
            texttemplate="%{text}",
            textfont={"size": 12},
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800,
        width=950,
        title=f"Airgram for {date.strftime('%Y-%m-%d')}, lat/lon: {st.session_state.target_latitude:.2f}, {st.session_state.target_longitude:.2f}",
        # xaxis=dict(title="Time"),
        yaxis=dict(title="Altitude (m)"),
        xaxis2=dict(title="Time", tickangle=-45),
        yaxis2=dict(title="Altitude (m)", range=[0, 3000]),
    )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig


def show_forecast():
    df_forecast = load_data()
    st.title("Termikkvarsel")
    st.markdown(
        f"Weather forecast from met.no's MEPS model. Current forecast is generated **{df_forecast['forecast_timestamp'][0]}**"
    )

    # Initialize query parameters on the first load
    if "query_initialized" not in st.session_state:
        initialize_query_params(df_forecast=df_forecast)
        st.session_state.query_initialized = True

    date_controls(df_forecast)

    with st.expander("Map", expanded=True):
        map_fig = build_map(
            df_forecast,
            selected_lat=st.session_state.target_latitude,  # Newly added
            selected_lon=st.session_state.target_longitude,  # Newly added
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
            print(selected_points)
            point = selected_points[0]
            new_lat = point["lat"]
            new_lon = point["lon"]
            print("Updated lat lon")

            if (
                st.session_state.target_latitude != new_lat
                or st.session_state.target_longitude != new_lon
            ):
                st.session_state.target_latitude = new_lat
                st.session_state.target_longitude = new_lon
                update_query_params(new_lat, new_lon)
                st.rerun()  # To reflect changes in query params

    if st.session_state.target_latitude is not None:
        wind_fig = create_daily_thermal_and_wind_airgram(
            df_forecast,
            lat=st.session_state.target_latitude,
            lon=st.session_state.target_longitude,
            date=st.session_state.forecast_date,
        )
        st.plotly_chart(wind_fig)
        plt.close()

    with st.expander("More settings", expanded=False):
        st.session_state.altitude_max = st.number_input(
            "Max altitude", 0, 4000, 3000, step=500
        )

    if st.session_state.target_latitude is not None:
        st.markdown("---")
        with st.expander("Sounding", expanded=False):
            st.title("SOUNDING IS NOT FIXED YET")
            date = datetime.datetime.combine(
                st.session_state.forecast_date, st.session_state.forecast_time
            )

            with st.spinner("Building sounding..."):
                sounding_fig = create_sounding(
                    df_forecast,
                    date=date.date(),
                    hour=date.hour,
                    altitude_max=st.session_state.altitude_max,
                    lon=st.session_state.target_longitude,
                    lat=st.session_state.target_latitude,
                )
            st.pyplot(sounding_fig)
            plt.close()

    st.markdown(
        "Wind and sounding data from MEPS model (main model used by met.no), including the estimated ground temperature. I've probably made many errors in this process."
    )


if __name__ == "__main__":
    run_streamlit = True
    if run_streamlit:
        st.set_page_config(page_title="Termikkvarsel", page_icon="🪂", layout="wide")
        show_forecast()
