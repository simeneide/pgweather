from __future__ import annotations

import datetime as dt
from pathlib import Path
from urllib.parse import parse_qs, urlencode
from zoneinfo import ZoneInfo

from dash import Dash, Input, Output, State, dcc, html, no_update

from . import forecast_service

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_TZ = ZoneInfo("Europe/Oslo")


def _to_iso(value: dt.datetime) -> str:
    return value.isoformat()


def _from_iso(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _to_local(value: dt.datetime) -> dt.datetime:
    return value.astimezone(LOCAL_TZ)


def _group_times_by_day(
    available_times: list[dt.datetime],
) -> dict[str, list[dt.datetime]]:
    """Group UTC datetimes by local date. Returns {iso_date_str: [utc_datetimes]}."""
    keyed = sorted(available_times, key=lambda t: _to_local(t))
    result: dict[str, list[dt.datetime]] = {}
    for t in keyed:
        day_key = _to_local(t).date().isoformat()
        result.setdefault(day_key, []).append(t)
    return result


def _day_label(iso_date: str) -> str:
    """E.g. '2026-02-13' -> 'Fri 13'."""
    d = dt.date.fromisoformat(iso_date)
    return d.strftime("%a %d")


def create_dash_app() -> Dash:
    app = Dash(
        __name__,
        title="\U0001fa82 Termikkvarselet",
        requests_pathname_prefix="/",
        suppress_callback_exceptions=True,
        assets_folder=str(_PROJECT_ROOT / "assets"),
    )

    df = forecast_service.load_forecast_data()
    available_times = forecast_service.get_available_times(df)
    if not available_times:
        raise RuntimeError("No forecast data available in detailed_forecasts")

    location_options = forecast_service.get_takeoff_options(df)
    if not location_options:
        raise RuntimeError("No takeoff locations available")
    names = [option["value"] for option in location_options]
    default_name = names[0]
    default_time = forecast_service.get_default_selected_time(df)

    # Group available datetimes by local day
    days_map = _group_times_by_day(available_times)
    day_keys = list(days_map.keys())

    default_day = _to_local(default_time).date().isoformat()
    if default_day not in day_keys:
        default_day = day_keys[0]

    # Pre-select the closest time within the default day
    day_times = days_map[default_day]
    default_time_iso = _to_iso(
        min(day_times, key=lambda t: abs((t - default_time).total_seconds()))
    )

    # Build day radio options
    day_radio_options = [{"label": _day_label(dk), "value": dk} for dk in day_keys]

    # Serialize days_map for client-side use
    days_map_serialized = {dk: [_to_iso(t) for t in ts] for dk, ts in days_map.items()}

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            # Stores
            dcc.Store(id="days-map-store", data=days_map_serialized),
            dcc.Store(id="selected-time-store", data=default_time_iso),
            html.H1("\U0001fa82 Termikkvarselet"),
            html.Div(
                [
                    # Day selector (RadioItems styled as pill buttons)
                    html.Label("Date", style={"fontWeight": 600}),
                    dcc.RadioItems(
                        id="day-radio",
                        options=day_radio_options,
                        value=default_day,
                        inline=True,
                        className="pill-radio",
                    ),
                    # Location
                    html.Label(
                        "Location",
                        style={"fontWeight": 600, "marginTop": "4px"},
                    ),
                    dcc.Dropdown(
                        id="location-dropdown",
                        options=location_options,
                        value=default_name,
                        clearable=False,
                    ),
                    # Advanced settings
                    html.Details(
                        [
                            html.Summary(
                                "Advanced map settings",
                                style={"cursor": "pointer", "fontWeight": 600},
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Map layer", style={"marginTop": "10px"}
                                    ),
                                    dcc.RadioItems(
                                        id="layer-radio",
                                        options=[
                                            {"label": "Both", "value": "both"},
                                            {"label": "Takeoffs", "value": "takeoffs"},
                                            {"label": "Areas", "value": "areas"},
                                        ],
                                        value="both",
                                        inline=True,
                                    ),
                                    html.Label("Map zoom"),
                                    dcc.Slider(
                                        id="zoom-slider",
                                        min=4,
                                        max=10,
                                        step=1,
                                        value=6,
                                    ),
                                    html.Label("Max altitude (m)"),
                                    dcc.Slider(
                                        id="altitude-slider",
                                        min=1000,
                                        max=4000,
                                        step=500,
                                        value=3000,
                                    ),
                                ],
                                style={"padding": "6px 4px"},
                            ),
                        ],
                        open=False,
                        style={
                            "border": "1px solid #d9dee6",
                            "borderRadius": "8px",
                            "padding": "8px",
                            "background": "#fafbfd",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gap": "10px",
                    "maxWidth": "860px",
                    "marginBottom": "12px",
                    "padding": "12px",
                    "background": "#f5f8fb",
                    "border": "1px solid #d9dee6",
                    "borderRadius": "12px",
                },
            ),
            dcc.Graph(id="map-graph", config={"displayModeBar": False}),
            # Windgram with Yr icons embedded — tap to select hour
            dcc.Graph(
                id="airgram-graph",
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                    "responsive": True,
                },
                style={"marginTop": "6px"},
            ),
            # Summary / forecast info at the bottom
            html.Div(
                id="summary-text",
                style={
                    "margin": "10px 0 0 0",
                    "color": "#888",
                    "fontSize": "12px",
                },
            ),
            # Attribution footer
            html.Footer(
                [
                    html.Div(
                        [
                            html.Span("Hosted and maintained by "),
                            html.A(
                                "eide.ai",
                                href="https://eide.ai",
                                target="_blank",
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            html.Span("Data: "),
                            html.A(
                                "MET Norway / MEPS",
                                href="https://www.met.no/",
                                target="_blank",
                            ),
                            html.Span(" | Weather symbols: "),
                            html.A(
                                "Yr",
                                href="https://www.yr.no/",
                                target="_blank",
                            ),
                            html.Span(" | Map: "),
                            html.A(
                                "OpenStreetMap",
                                href="https://www.openstreetmap.org/copyright",
                                target="_blank",
                            ),
                        ],
                        style={"marginTop": "4px"},
                    ),
                ],
                style={
                    "marginTop": "24px",
                    "paddingTop": "12px",
                    "borderTop": "1px solid #e2e8f0",
                    "fontSize": "11px",
                    "color": "#999",
                    "textAlign": "center",
                },
            ),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    )

    # -------------------------------------------------------------------
    # Day changed -> pick best time in new day and update store
    # -------------------------------------------------------------------
    @app.callback(
        Output("selected-time-store", "data", allow_duplicate=True),
        Input("day-radio", "value"),
        State("selected-time-store", "data"),
        State("days-map-store", "data"),
        prevent_initial_call=True,
    )
    def on_day_changed(day_key, current_time_iso, days_map_data):
        time_isos = days_map_data.get(day_key, [])
        if not time_isos:
            return current_time_iso

        if current_time_iso:
            current_hour = _to_local(_from_iso(current_time_iso)).hour
        else:
            current_hour = 12

        best_iso = time_isos[0]
        best_diff = 999
        for iso in time_isos:
            h = _to_local(_from_iso(iso)).hour
            diff = abs(h - current_hour)
            if diff < best_diff:
                best_diff = diff
                best_iso = iso

        return best_iso

    # -------------------------------------------------------------------
    # Windgram click -> select time for map
    # -------------------------------------------------------------------
    @app.callback(
        Output("selected-time-store", "data", allow_duplicate=True),
        Input("airgram-graph", "clickData"),
        State("selected-time-store", "data"),
        State("days-map-store", "data"),
        State("day-radio", "value"),
        prevent_initial_call=True,
    )
    def on_windgram_clicked(click_data, current_time_iso, days_map_data, day_key):
        if not click_data or "points" not in click_data:
            return no_update
        point = click_data["points"][0]
        x_label = point.get("x")  # e.g. "12h"
        if not x_label or not isinstance(x_label, str) or not x_label.endswith("h"):
            return no_update

        try:
            clicked_hour = int(x_label.replace("h", ""))
        except ValueError:
            return no_update

        # Find the UTC time matching this local hour in the current day
        time_isos = days_map_data.get(day_key, [])
        for iso in time_isos:
            if _to_local(_from_iso(iso)).hour == clicked_hour:
                return iso

        return current_time_iso

    # -------------------------------------------------------------------
    # Location from URL query param
    # -------------------------------------------------------------------
    @app.callback(
        Output("location-dropdown", "value", allow_duplicate=True),
        Input("url", "search"),
        State("location-dropdown", "value"),
        prevent_initial_call=True,
    )
    def location_from_query(search: str, current_value: str):
        if not search:
            return current_value
        params = parse_qs(search.lstrip("?"))
        location = params.get("location", [None])[0]
        if location and location in names:
            return location
        return current_value

    # -------------------------------------------------------------------
    # Location -> URL query param
    # -------------------------------------------------------------------
    @app.callback(
        Output("url", "search"),
        Input("location-dropdown", "value"),
        State("url", "search"),
        prevent_initial_call=True,
    )
    def query_from_location(selected_name: str, current_search: str):
        params = parse_qs((current_search or "").lstrip("?"))
        params["location"] = [selected_name]
        return "?" + urlencode(params, doseq=True)

    # -------------------------------------------------------------------
    # Main figures callback — driven by store + controls
    # -------------------------------------------------------------------
    @app.callback(
        Output("map-graph", "figure"),
        Output("airgram-graph", "figure"),
        Output("summary-text", "children"),
        Input("selected-time-store", "data"),
        Input("location-dropdown", "value"),
        Input("layer-radio", "value"),
        Input("zoom-slider", "value"),
        Input("altitude-slider", "value"),
    )
    def update_figures(
        selected_time_iso: str,
        selected_name: str,
        map_layer: str,
        zoom: int,
        altitude_max: int,
    ):
        if not selected_time_iso:
            selected_time_iso = _to_iso(forecast_service.get_default_selected_time())
        selected_time_utc = _from_iso(selected_time_iso)
        selected_time_local = _to_local(selected_time_utc)
        df_now = forecast_service.load_forecast_data()

        map_fig = forecast_service.build_map_figure(
            selected_time=selected_time_utc,
            selected_name=selected_name,
            map_layer=map_layer,
            zoom=zoom,
            df=df_now,
        )

        # Fetch Yr data synced to MEPS hours
        day = selected_time_local.date()
        forecast_hours = forecast_service.get_forecast_hours_for_day(
            selected_name, day, df_now
        )
        yr_entries = forecast_service.get_yr_weather_for_day(
            selected_name, day, df_now, restrict_to_hours=forecast_hours
        )

        airgram_fig = forecast_service.build_airgram_figure(
            target_name=selected_name,
            selected_date=day,
            altitude_max=altitude_max,
            yr_entries=yr_entries,
            selected_hour=selected_time_local.hour,
            df=df_now,
        )

        summary = forecast_service.get_summary(selected_name, selected_time_utc, df_now)
        latest_ts = forecast_service.get_latest_forecast_timestamp(df_now)
        age_hours = (
            dt.datetime.now(dt.timezone.utc) - latest_ts
        ).total_seconds() / 3600
        merged = (
            f"{summary} | Forecast updated "
            f"{_to_local(latest_ts).strftime('%Y-%m-%d %H:%M')}"
            f" local ({age_hours:.1f}h ago)"
        )

        return map_fig, airgram_fig, merged

    # -------------------------------------------------------------------
    # Click map -> select location
    # -------------------------------------------------------------------
    @app.callback(
        Output("location-dropdown", "value", allow_duplicate=True),
        Input("map-graph", "clickData"),
        State("location-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_location_from_map(click_data, current_value):
        if (
            not click_data
            or "points" not in click_data
            or len(click_data["points"]) == 0
        ):
            return current_value
        point = click_data["points"][0]
        name = point.get("customdata")
        return name or current_value

    return app
