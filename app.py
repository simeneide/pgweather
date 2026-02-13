import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import datetime
from utils import interpolate_color
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import db_utils
import polars as pl
import json
import pytz


def update_session_and_query_parameters(**kwargs):
    """Update or initialize session and query parameters, allowing for overriding via kwargs."""

    def _get_query_param(name):
        value = st.query_params.get(name)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _parse_selected_timestamp(value):
        if not value:
            return None
        try:
            parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            return parsed.astimezone(datetime.timezone.utc)
        except ValueError:
            return None

    # Default values
    default_values = {
        "target_name": "Barten",
        "target_latitude": 61.2479,
        "target_longitude": 7.08998,
        "selected_timestamp": datetime.datetime.now(datetime.timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        + datetime.timedelta(hours=24),
        "altitude_max": 3500,
        "zoom": 6,  # Default zoom level
    }

    # Initialize or update session state with query parameters, defaults, or overrides from kwargs
    for key, default_value in default_values.items():
        assigned = False
        if key in kwargs:
            st.session_state[key] = kwargs[key]
            assigned = True
        elif key == "target_name":
            query_value = _get_query_param("target_name")
            if query_value:
                st.session_state[key] = query_value
                assigned = True
        elif key == "selected_timestamp":
            query_value = _parse_selected_timestamp(
                _get_query_param("selected_timestamp")
            )
            if query_value:
                st.session_state[key] = query_value
                assigned = True
        elif key in ["altitude_max", "zoom"]:
            query_value = _get_query_param(key)
            if query_value is not None:
                try:
                    st.session_state[key] = int(query_value)
                    assigned = True
                except ValueError:
                    pass

        if (not assigned) and (key not in st.session_state):
            st.session_state[key] = default_value

    # Update the streamlit query parameters from session state
    st.query_params.update(
        {
            "target_name": st.session_state.target_name,
            "selected_timestamp": st.session_state.selected_timestamp.isoformat(),
            "altitude_max": str(st.session_state.altitude_max),
            "zoom": str(st.session_state.zoom),
        }
    )


def align_selected_timestamp_to_data(df_forecast_detailed):
    selected = st.session_state.get("selected_timestamp")
    if selected is None:
        return

    if selected.tzinfo is None:
        selected = selected.replace(tzinfo=datetime.timezone.utc)

    available_times = df_forecast_detailed.get_column("time").unique().sort().to_list()
    if not available_times:
        return

    if selected in available_times:
        return

    nearest = min(available_times, key=lambda t: abs((t - selected).total_seconds()))
    st.session_state.selected_timestamp = nearest


@st.cache_resource(ttl=3600)
def load_data(forecast_type="detailed"):
    """
    Connects to the database and loads the forecast data as a Polars DataFrame.
    forecast_type = "detailed"


    # For dev:
    # Save to local file
    df_forecast_detailed.write_parquet("tmp_weather_forecasts.parquet")

    # load from local file
    df_forecast_detailed = pl.read_parquet("tmp_weather_forecasts.parquet")
    """
    assert forecast_type in ["detailed", "area"], (
        "forecast_type must be either 'detailed' or 'area'"
    )
    with st.spinner("Loading forecast from database...", show_time=True):
        db = db_utils.Database()

        # Read the data from the database
        query = f"""
        SELECT *
        FROM {forecast_type}_forecasts
        """
        # forecast_timestamp, time, name, altitude, air_temperature_ml, x_wind_ml, y_wind_ml, longitude, latitude, wind_speed, thermal_temp_diff, thermal_top, thermal_height_above_ground

        df_forecast_detailed = db.read(query)  # Using the Polars DataFrame
        df_forecast_detailed = df_forecast_detailed.with_columns(
            [
                pl.col("forecast_timestamp")
                .cast(pl.Datetime)
                .dt.replace_time_zone(
                    "UTC"
                ),  # .dt.convert_time_zone("Europe/Brussels"),
                pl.col("time")
                .cast(pl.Datetime)
                .dt.replace_time_zone(
                    "UTC"
                ),  # .dt.convert_time_zone("Europe/Brussels"),
            ]
        )

    return df_forecast_detailed


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


def date_controls(df_forecast_detailed):
    # @st.cache_data(ttl=3600)
    def get_forecast_days(df_forecast_detailed):
        start_stop_time = [
            df_forecast_detailed.get_column("time").min(),
            df_forecast_detailed.get_column("time").max()
            - datetime.timedelta(hours=12),
        ]
        today = datetime.datetime.now().date()

        # Generate available days within the dataset's time range
        available_days = pd.date_range(
            start=start_stop_time[0], end=start_stop_time[1]
        ).date
        return available_days, today

    available_days, _ = get_forecast_days(df_forecast_detailed)

    cet = pytz.timezone("Europe/Oslo")
    utc = pytz.timezone("UTC")
    selected_timestamp = st.session_state.get("selected_timestamp")
    if selected_timestamp.tzinfo is None:
        selected_timestamp = selected_timestamp.replace(tzinfo=utc)
    selected_time_cet = selected_timestamp.astimezone(cet)

    selected_day = st.selectbox(
        "Day",
        options=available_days,
        index=max(0, list(available_days).index(selected_time_cet.date()))
        if selected_time_cet.date() in available_days
        else 0,
        format_func=lambda day: day.strftime("%a %d %b"),
    )

    quick_cols = st.columns(3)
    quick_hours = {"Morning": 9, "Noon": 12, "Evening": 18}
    selected_hour = selected_time_cet.hour
    for idx, label in enumerate(quick_hours):
        with quick_cols[idx]:
            if st.button(label, use_container_width=True):
                selected_hour = quick_hours[label]

    selected_hour = st.slider(
        "Hour (local)",
        min_value=0,
        max_value=23,
        value=selected_hour,
        format="%02d:00",
    )

    cet_time = cet.localize(
        datetime.datetime.combine(selected_day, datetime.time(selected_hour, 0))
    )

    # Convert it to UTC
    utc_time = cet_time.astimezone(utc)

    # Update the forecast time with the selected hour from the slider
    if utc_time != st.session_state.get("selected_timestamp"):
        st.session_state.selected_timestamp = utc_time
        st.rerun()

    # st.write(
    #     f"Selected time: {selected_time_cet.strftime('%Y-%m-%d %H:%M')} CET, {utc_time.strftime('%Y-%m-%d %H:%M')} UTC"
    # )


@st.cache_data(ttl=1800)
def build_map(
    df_forecast_detailed,
    selected_lat=None,
    selected_lon=None,
    selected_timestamp=None,
    show_area=True,
    show_detailed=True,
):
    """
    date = datetime.datetime.now().replace(minute=0, second=0, microsecond=0).date()
    hour = datetime.datetime.now().replace(minute=0, second=0, microsecond=0).hour
    hour = datetime.time(hour, 0, 0)
    selected_timestamp = df_forecast_detailed.get_column("time").min()
    selected_lon = df_forecast_detailed.get_column("longitude").to_numpy()[0]
    selected_lat = df_forecast_detailed.get_column("latitude").to_numpy()[0]
    """

    ## BUILD AREA MAP
    with open("Kommuner-S.geojson", "r") as file:
        areas = json.load(file)
    subset_area = df_forecast_detailed.filter(
        (pl.col("time") == selected_timestamp), pl.col("point_type") == "area"
    )

    name_values = subset_area.get_column("name").to_numpy()
    thermal_top_area_values = subset_area.get_column("thermal_top").to_numpy().round()

    # Define a custom colorscale
    thermal_colorscale = [
        (0.0, "grey"),  # 0
        (0.2, "yellow"),  # 1k
        (0.4, "red"),  # 2k
        (0.6, "violet"),  # 3k
        # (0.8, "darkred"), # 4k
        (1.0, "black"),  # 5k
    ]

    fig = go.Figure()
    if show_area:
        area_map = go.Choroplethmap(
            geojson=areas,
            zmin=0,
            zmax=5000,
            featureidkey="properties.name",
            locations=name_values,
            ids=name_values,
            z=thermal_top_area_values,
            colorscale=thermal_colorscale,
            showscale=True,
            hovertext=[
                f"{name} | median thermal top: {ht} m"
                for ht, name in zip(thermal_top_area_values, name_values)
            ],
            marker_opacity=0.35,
            colorbar=dict(title="Thermal top (m)"),
        )
        fig.add_trace(area_map)
    ## BUILD DETAILED MAP
    if show_detailed:
        subset = df_forecast_detailed.filter(
            (pl.col("time") == selected_timestamp), pl.col("point_type") != "area"
        )

        latitude_values = subset.get_column("latitude").to_numpy()
        longitude_values = subset.get_column("longitude").to_numpy()
        thermal_top_values = subset.get_column("thermal_top").to_numpy().round()
        name_values = subset.get_column("name").to_numpy()

        # Determine whether a point is selected
        selected_points = (latitude_values == selected_lat) & (
            longitude_values == selected_lon
        )

        # Use conditional logic to define marker properties
        marker_size = np.where(selected_points, 20, 9)  # Larger size for selected point

        detailed_map = go.Scattermap(
            lat=latitude_values,
            lon=longitude_values,
            mode="markers",
            line=dict(width=2, color="grey"),
            marker=go.scattermap.Marker(
                size=marker_size,
                cmin=0,
                cmax=5000,
                color=thermal_top_values,
                colorscale=thermal_colorscale,
                opacity=1,
                showscale=False,
                colorbar=dict(title="Thermal Height (m)"),
            ),
            ids=name_values,
            text=[
                f"{name} | thermal top: {ht} m"
                for ht, name in zip(thermal_top_values, name_values)
            ],
            hoverinfo="text",
        )

        fig.add_trace(detailed_map)

    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            center=dict(lat=selected_lat, lon=selected_lon), zoom=st.session_state.zoom
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig


# @st.cache_data(ttl=3600)
def create_daily_thermal_and_wind_airgram(df_forecast_detailed, target_name, date):
    """
    Create a Plotly subplot figure for a single day's thermal and wind data.
    The top subplot shows wind data as arrows for direction and color for strength.
    The bottom subplot shows thermal temperature differences.


    target_name = "Barten" # st.session_state.target_name

    # Visualize point
    import plotly.express as px
    fig = px.scatter_mapbox(
        lat=[lat],
        lon=[lon],
        zoom=10,  # Adjust the zoom level as needed
        center={'lat': lat, 'lon': lon},
        mapbox_style='open-street-map',  # Use the same map style as in the build_map function
        height=400,
        width=600
    )

    fig.update_traces(marker=dict(size=10, color="red"))  # Customize marker appearance
    fig.update_layout(title="Location Marker", margin={"r":0,"t":0,"l":0,"b":0})

    """

    display_start_hour = 8
    display_end_hour = 21
    location_data = df_forecast_detailed.with_columns(
        time=pl.col("time").dt.convert_time_zone("Europe/Oslo")
    ).filter(
        # Ensure correct date
        (pl.col("time").dt.date() == date),
        # Ensure correct hours
        (pl.col("time").dt.hour().is_between(display_start_hour, display_end_hour)),
        pl.col("name") == target_name,
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
        .with_columns(
            wind_direction=-pl.arctan2("y_wind_ml", "x_wind_ml").degrees() + 90
        )
        .sort("time")
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.05,
        subplot_titles=(
            "Wind Speed and Direction [m/s]",
            "Thermal Temperature Difference [°C]",
        ),
    )

    ## WIND PLOT
    wind_altitues = np.arange(200.0, 3000.0, 400)
    plot_frame_wind = plot_frame.sort("time", "altitude").filter(
        pl.col("altitude").is_in(wind_altitues)
    )

    # Add arrows
    fig.add_trace(
        go.Scatter(
            x=plot_frame_wind.select("time").to_series().to_list(),
            y=plot_frame_wind.select("altitude").to_numpy().squeeze(),
            mode="markers",
            marker=dict(
                symbol="arrow",
                size=20,
                angle=plot_frame_wind.select("wind_direction").to_numpy().squeeze(),
                color=[
                    interpolate_color(s)
                    for s in plot_frame_wind.select("wind_speed").to_numpy().squeeze()
                ],
                showscale=False,
                cmin=0,
                cmax=20,
            ),
            hoverinfo="text",
            text=[
                f"Alt: {alt} m, Speed: {spd:.1f} m/s, Direction: {angle:.1f}°"
                for alt, spd, angle in zip(
                    plot_frame_wind.select("altitude").to_numpy().squeeze(),
                    plot_frame_wind.select("wind_speed").to_numpy().squeeze(),
                    plot_frame_wind.select("wind_direction").to_numpy().squeeze(),
                )
            ],
        ),
        row=1,
        col=1,
    )

    # Add wind speed numbers next to the arrows
    fig.add_trace(
        go.Scatter(
            x=plot_frame_wind.select("time").to_series().to_list(),
            y=plot_frame_wind.select("altitude").to_numpy().squeeze(),
            mode="text",
            text=[
                f"{spd:.1f}"
                for spd in plot_frame_wind.select("wind_speed").to_numpy().squeeze()
            ],
            textposition="middle right",
            textfont=dict(color="black", size=11, family="Arial"),
            showlegend=False,
            hoverinfo="skip",
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
            x=plot_frame.select("time").to_series().to_list(),
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
        title=f"Airgram for {plot_frame.row(0, named=True)['name']} on {date.strftime('%A')} {date.strftime('%Y-%m-%d')}",
        yaxis=dict(title="Altitude (m)"),
        xaxis2=dict(title="Time", tickangle=-45),
        yaxis2=dict(title="Altitude (m)", range=[0, 3000]),
    )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig


def main():
    df_forecast_detailed = load_data("detailed")
    st.title("Termikkvarsel")

    latest_ts = df_forecast_detailed.get_column("forecast_timestamp").max()
    age_hours = (
        datetime.datetime.now(datetime.timezone.utc) - latest_ts
    ).total_seconds() / 3600
    if age_hours > 8:
        st.warning(
            f"Forecast last updated {latest_ts} UTC ({age_hours:.1f} hours ago).",
            icon="⏱️",
        )
    else:
        st.caption(f"Forecast updated {latest_ts} UTC ({age_hours:.1f} hours ago)")

    update_session_and_query_parameters()
    align_selected_timestamp_to_data(df_forecast_detailed)

    with st.sidebar:
        st.subheader("Controls")
        st.session_state.zoom = st.slider("Map zoom", 4, 10, st.session_state.zoom)
        map_layer = st.radio(
            "Map layers",
            options=["Both", "Takeoffs", "Areas"],
            horizontal=False,
        )
        st.session_state.altitude_max = st.number_input(
            "Max altitude (m)",
            0,
            4000,
            st.session_state.altitude_max,
            step=500,
        )
        date_controls(df_forecast_detailed)

    with st.expander("Map", expanded=True):
        map_fig = build_map(
            df_forecast_detailed,
            selected_lat=st.session_state.target_latitude,
            selected_lon=st.session_state.target_longitude,
            selected_timestamp=st.session_state.selected_timestamp,
            show_area=map_layer in ["Both", "Areas"],
            show_detailed=map_layer in ["Both", "Takeoffs"],
        )

        def a_callback():
            selected_points = (
                st.session_state.get("map_selection").get("selection").get("points", [])
            )
            if len(selected_points) > 0:
                point = selected_points[0]
                if point.get("ct"):
                    point["lon"] = point.get("ct")[0]
                    point["lat"] = point.get("ct")[1]
                point["name"] = point.get("id")
                update_session_and_query_parameters(
                    target_latitude=point["lat"],
                    target_longitude=point["lon"],
                    target_name=point["name"],
                )

        st.plotly_chart(
            map_fig,
            key="map_selection",
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": False},
            on_select=a_callback,
        )

    selected_row = (
        df_forecast_detailed.filter(
            (pl.col("point_type") != "area")
            & (pl.col("name") == st.session_state.target_name)
            & (pl.col("time") == st.session_state.selected_timestamp)
            & (pl.col("altitude") <= 1000)
        )
        .sort("altitude", descending=True)
        .head(1)
    )
    if len(selected_row) > 0:
        st.info(
            f"Selected: {selected_row[0, 'name']} | Time: {st.session_state.selected_timestamp} UTC | "
            f"Thermal top: {selected_row[0, 'thermal_top']:.0f} m | "
            f"Wind: {selected_row[0, 'wind_speed']:.1f} m/s",
            icon="🪂",
        )

    if st.session_state.target_name is not None:
        wind_fig = create_daily_thermal_and_wind_airgram(
            df_forecast_detailed,
            target_name=st.session_state.target_name,
            date=st.session_state.selected_timestamp.date(),
        )

        st.plotly_chart(
            wind_fig,
            use_container_width=True,
            config={"scrollZoom": False, "displayModeBar": False, "staticPlot": False},
        )

    st.markdown(
        f""" \
# Om appen

- Dette varselet er generert **{df_forecast_detailed["forecast_timestamp"][0]}**.
- Værvarselet er hentet fra Meteorlogisk institutt sin MEPS modell. 
- Dette er en prototype for å vise hvordan "yr fungerer i høyden".
- Varselet er generert for en rekke paragliderstarter og kommuner. 
- På kommunenivå er varselet medianen av alle punktene i kommunen.
- Airgram viser temperaturforskjellen mellom lufta og bakken på samme punkt, begge estimert fra modellen. Har ikke full kontroll på hvordan bakketemperaturen beregnes, men det er noe terrengmodell der, feks blir det ingen temperaturforskjell i fjorden.
- Garantert noen feil her, jeg tar ingen ansvar for tidlige landinger!
- Gi gjerne feedback, eller enda bedre, bidra! [Github repo](https://github.com/simeneide/pgweather)
- Laget av Simen Eide
"""
    )


if __name__ == "__main__":
    run_streamlit = True
    if run_streamlit:
        st.set_page_config(page_title="Termikkvarsel", page_icon="🪂", layout="wide")
        main()
