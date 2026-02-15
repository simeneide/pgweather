from __future__ import annotations

import datetime as dt
from pathlib import Path
from urllib.parse import parse_qs, urlencode
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

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


def _empty_airgram() -> go.Figure:
    """Placeholder figure shown when no location is selected."""
    fig = go.Figure()
    fig.add_annotation(
        text="Select a takeoff on the map to see the windgram.",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_layout(
        height=200,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _compute_layout_defaults() -> dict[str, object]:
    """Compute initial layout values from the forecast metadata.

    Uses the lightweight ``load_metadata()`` query — no full DataFrame load.
    Called once per page load (via the function-based layout) so the app can
    start listening immediately without waiting for the database.
    """
    meta = forecast_service.load_metadata()
    available_times = meta.available_times
    if not available_times:
        raise RuntimeError("No forecast data available in detailed_forecasts")

    location_options = forecast_service.get_takeoff_options()
    if not location_options:
        raise RuntimeError("No takeoff locations available")
    names = [option["value"] for option in location_options]
    default_time = forecast_service.get_default_selected_time()

    days_map = _group_times_by_day(available_times)
    day_keys = list(days_map.keys())

    default_day = _to_local(default_time).date().isoformat()
    if default_day not in day_keys:
        default_day = day_keys[0]

    target_hour = 14
    day_times = days_map[default_day]
    default_time_iso = _to_iso(
        min(day_times, key=lambda t: abs(_to_local(t).hour - target_hour))
    )

    day_radio_options = [{"label": _day_label(dk), "value": dk} for dk in day_keys]
    days_map_serialized = {dk: [_to_iso(t) for t in ts] for dk, ts in days_map.items()}

    return {
        "location_options": location_options,
        "names": names,
        "default_day": default_day,
        "default_time_iso": default_time_iso,
        "day_radio_options": day_radio_options,
        "days_map_serialized": days_map_serialized,
        "day_keys": day_keys,
    }


def create_dash_app() -> Dash:
    app = Dash(
        __name__,
        title="Termikkvarselet",
        update_title=None,
        requests_pathname_prefix="/",
        suppress_callback_exceptions=True,
        assets_folder=str(_PROJECT_ROOT / "assets"),
    )
    app._favicon = "logo.png"

    # ---------------------------------------------------------------
    # Function-based layout: defers DB access to first page request
    # so uvicorn can start listening immediately.
    # ---------------------------------------------------------------
    def serve_layout() -> html.Div:
        defaults = _compute_layout_defaults()
        location_options = defaults["location_options"]
        default_day = defaults["default_day"]
        default_time_iso = defaults["default_time_iso"]
        day_radio_options = defaults["day_radio_options"]
        days_map_serialized = defaults["days_map_serialized"]
        day_keys = defaults["day_keys"]

        return html.Div(
            [
                dcc.Location(id="url", refresh=False),
                # Stores
                dcc.Store(id="days-map-store", data=days_map_serialized),
                dcc.Store(id="day-keys-store", data=day_keys),
                dcc.Store(id="selected-time-store", data=default_time_iso),
                dcc.Store(id="modal-open-store", data=False),
                # Hidden keyboard listener
                html.Div(
                    id="keyboard-listener",
                    tabIndex="0",
                    style={
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                        "height": "100%",
                        "zIndex": -1,
                        "opacity": 0,
                    },
                    **{"data-dummy": ""},
                ),
                html.H1("Termikkvarselet"),
                # Map controls panel
                html.Div(
                    [
                        # Day selector (also mirrored inside the modal)
                        dcc.RadioItems(
                            id="day-radio",
                            options=day_radio_options,
                            value=default_day,
                            inline=True,
                            className="pill-radio",
                        ),
                        # Hour +/- controls with current time display
                        html.Div(
                            [
                                html.Button(
                                    "\u25c0",
                                    id="hour-prev-btn",
                                    n_clicks=0,
                                    className="hour-btn",
                                    title="Previous hour",
                                ),
                                html.Span(
                                    id="current-hour-label",
                                    style={
                                        "minWidth": "56px",
                                        "textAlign": "center",
                                        "fontWeight": 600,
                                        "fontSize": "14px",
                                        "color": "#1e293b",
                                    },
                                ),
                                html.Button(
                                    "\u25b6",
                                    id="hour-next-btn",
                                    n_clicks=0,
                                    className="hour-btn",
                                    title="Next hour",
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "6px",
                            },
                        ),
                        # Location search
                        dcc.Dropdown(
                            id="location-dropdown",
                            options=location_options,
                            value=None,
                            clearable=True,
                            placeholder="Search takeoff...",
                        ),
                        # Hidden stores for removed controls (keep callback inputs valid)
                        dcc.Store(id="layer-radio", data="both"),
                        dcc.Store(id="zoom-slider", data=6),
                    ],
                    style={
                        "display": "grid",
                        "gap": "8px",
                        "maxWidth": "860px",
                        "marginBottom": "8px",
                        "padding": "10px 12px",
                        "background": "#f5f8fb",
                        "border": "1px solid #d9dee6",
                        "borderRadius": "12px",
                    },
                ),
                dcc.Graph(
                    id="map-graph",
                    config={"displayModeBar": False},
                    style={"height": "75vh", "minHeight": "350px"},
                ),
                # Summary / forecast info below map (clickable to reopen modal)
                html.Div(
                    id="summary-text",
                    n_clicks=0,
                    style={
                        "margin": "6px 0 0 0",
                        "color": "#888",
                        "fontSize": "12px",
                        "cursor": "pointer",
                    },
                    title="Click to open windgram",
                ),
                # --- Windgram modal overlay ---
                html.Div(
                    id="modal-wrapper",
                    className="modal-wrapper hidden",
                    children=[
                        html.Div(
                            id="modal-backdrop",
                            className="modal-backdrop-layer",
                            n_clicks=0,
                        ),
                        html.Div(
                            id="modal-content",
                            className="modal-content",
                            children=[
                                html.Button(
                                    "\u00d7",
                                    id="modal-close-btn",
                                    className="modal-close-btn",
                                    n_clicks=0,
                                ),
                                # Modal header: location name + date pills
                                html.Div(
                                    id="modal-header",
                                    className="modal-header",
                                    children=[
                                        html.H2(
                                            id="modal-title",
                                            children="",
                                            style={
                                                "margin": "0 0 6px 0",
                                                "fontSize": "1.1rem",
                                                "fontWeight": 700,
                                                "color": "#1e293b",
                                                "paddingRight": "36px",
                                            },
                                        ),
                                        # Day selector (synced with main page)
                                        dcc.RadioItems(
                                            id="modal-day-radio",
                                            options=day_radio_options,
                                            value=default_day,
                                            inline=True,
                                            className="pill-radio",
                                        ),
                                        # Fixed altitude (hidden)
                                        dcc.Store(
                                            id="altitude-slider",
                                            data=3000,
                                        ),
                                    ],
                                ),
                                # Windgram graph
                                dcc.Graph(
                                    id="airgram-graph",
                                    config={
                                        "displayModeBar": False,
                                        "scrollZoom": False,
                                        "doubleClick": False,
                                        "responsive": True,
                                    },
                                ),
                            ],
                        ),
                    ],
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

    app.layout = serve_layout

    # ===================================================================
    # Callbacks
    # ===================================================================

    # -------------------------------------------------------------------
    # Sync: main day-radio -> modal day-radio
    # -------------------------------------------------------------------
    @app.callback(
        Output("modal-day-radio", "value", allow_duplicate=True),
        Input("day-radio", "value"),
        prevent_initial_call=True,
    )
    def sync_day_to_modal(day_key):
        return day_key

    # -------------------------------------------------------------------
    # Sync: modal day-radio -> main day-radio
    # -------------------------------------------------------------------
    @app.callback(
        Output("day-radio", "value", allow_duplicate=True),
        Input("modal-day-radio", "value"),
        prevent_initial_call=True,
    )
    def sync_day_to_main(day_key):
        return day_key

    # -------------------------------------------------------------------
    # Day changed (either radio) -> pick best time in new day
    # -------------------------------------------------------------------
    @app.callback(
        Output("selected-time-store", "data", allow_duplicate=True),
        Input("day-radio", "value"),
        Input("modal-day-radio", "value"),
        State("selected-time-store", "data"),
        State("days-map-store", "data"),
        prevent_initial_call=True,
    )
    def on_day_changed(main_day, modal_day, current_time_iso, days_map_data):
        # Use whichever was triggered; both should be in sync
        triggered = callback_context.triggered_id
        day_key = modal_day if triggered == "modal-day-radio" else main_day

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
    # Location from URL query param (on initial load)
    # -------------------------------------------------------------------
    @app.callback(
        Output("location-dropdown", "value", allow_duplicate=True),
        Input("url", "search"),
        State("location-dropdown", "value"),
        prevent_initial_call=True,
    )
    def location_from_query(search: str, current_value: str | None):
        if not search:
            return no_update
        params = parse_qs(search.lstrip("?"))
        location = params.get("location", [None])[0]
        if location and location in forecast_service.get_takeoff_names():
            return location
        return no_update

    # -------------------------------------------------------------------
    # Location -> URL query param
    # -------------------------------------------------------------------
    @app.callback(
        Output("url", "search"),
        Input("location-dropdown", "value"),
        State("url", "search"),
        prevent_initial_call=True,
    )
    def query_from_location(selected_name: str | None, current_search: str):
        if not selected_name:
            return no_update
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
        Output("modal-title", "children"),
        Input("selected-time-store", "data"),
        Input("location-dropdown", "value"),
        Input("layer-radio", "data"),
        Input("zoom-slider", "data"),
        Input("altitude-slider", "data"),
    )
    def update_figures(
        selected_time_iso: str,
        selected_name: str | None,
        _map_layer: str,
        zoom: int,
        altitude_max: int,
    ):
        if not selected_time_iso:
            selected_time_iso = _to_iso(forecast_service.get_default_selected_time())
        selected_time_utc = _from_iso(selected_time_iso)
        selected_time_local = _to_local(selected_time_utc)

        map_fig = forecast_service.build_map_figure(
            selected_time=selected_time_utc,
            selected_name=selected_name,
            zoom=zoom,
        )

        # If no location selected, return empty airgram
        if not selected_name:
            return map_fig, _empty_airgram(), "", ""

        # Build airgram for selected location
        day = selected_time_local.date()
        forecast_hours = forecast_service.get_forecast_hours_for_day(selected_name, day)
        yr_entries = forecast_service.get_yr_weather_for_day(
            selected_name, day, restrict_to_hours=forecast_hours
        )

        airgram_fig = forecast_service.build_airgram_figure(
            target_name=selected_name,
            selected_date=day,
            altitude_max=altitude_max,
            yr_entries=yr_entries,
            selected_hour=selected_time_local.hour,
        )

        summary = forecast_service.get_summary(selected_name, selected_time_utc)
        latest_ts = forecast_service.get_latest_forecast_timestamp()
        age_hours = (
            dt.datetime.now(dt.timezone.utc) - latest_ts
        ).total_seconds() / 3600
        merged = (
            f"{summary} | Forecast updated "
            f"{_to_local(latest_ts).strftime('%Y-%m-%d %H:%M')}"
            f" local ({age_hours:.1f}h ago)"
        )

        # Modal title: location name + date
        date_label = selected_time_local.strftime("%a %d %b")
        modal_title = f"{selected_name} — {date_label}"

        return map_fig, airgram_fig, merged, modal_title

    # -------------------------------------------------------------------
    # Click map -> select location + open modal
    # -------------------------------------------------------------------
    @app.callback(
        Output("location-dropdown", "value", allow_duplicate=True),
        Output("modal-open-store", "data", allow_duplicate=True),
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
            return no_update, no_update
        point = click_data["points"][0]
        name = point.get("customdata")
        if not name:
            return no_update, no_update
        # Always open modal on map click (even if same location re-clicked)
        return name, True

    # -------------------------------------------------------------------
    # Open modal when a location is selected
    # -------------------------------------------------------------------
    @app.callback(
        Output("modal-open-store", "data", allow_duplicate=True),
        Input("location-dropdown", "value"),
        prevent_initial_call=True,
    )
    def open_modal_on_location(selected_name):
        if selected_name:
            return True
        return False

    # -------------------------------------------------------------------
    # Reopen modal when summary text is clicked
    # -------------------------------------------------------------------
    @app.callback(
        Output("modal-open-store", "data", allow_duplicate=True),
        Input("summary-text", "n_clicks"),
        State("location-dropdown", "value"),
        prevent_initial_call=True,
    )
    def reopen_modal_on_summary_click(n_clicks, selected_name):
        if n_clicks and selected_name:
            return True
        return no_update

    # -------------------------------------------------------------------
    # Close modal: close button or backdrop click
    # -------------------------------------------------------------------
    @app.callback(
        Output("modal-open-store", "data", allow_duplicate=True),
        Input("modal-close-btn", "n_clicks"),
        Input("modal-backdrop", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_modal(close_clicks, backdrop_clicks):
        return False

    # -------------------------------------------------------------------
    # Toggle modal visibility via className on the wrapper
    # -------------------------------------------------------------------
    @app.callback(
        Output("modal-wrapper", "className"),
        Input("modal-open-store", "data"),
    )
    def toggle_modal_visibility(is_open):
        if is_open:
            return "modal-wrapper"
        return "modal-wrapper hidden"

    # -------------------------------------------------------------------
    # Hour +/- buttons -> step through available times in the current day
    # -------------------------------------------------------------------
    @app.callback(
        Output("selected-time-store", "data", allow_duplicate=True),
        Input("hour-prev-btn", "n_clicks"),
        Input("hour-next-btn", "n_clicks"),
        State("selected-time-store", "data"),
        State("days-map-store", "data"),
        State("day-radio", "value"),
        prevent_initial_call=True,
    )
    def on_hour_button(
        prev_clicks, next_clicks, current_time_iso, days_map_data, day_key
    ):
        triggered = callback_context.triggered_id
        direction = -1 if triggered == "hour-prev-btn" else 1

        time_isos = days_map_data.get(day_key, [])
        if not time_isos or not current_time_iso:
            return no_update

        # Find current index
        try:
            idx = time_isos.index(current_time_iso)
        except ValueError:
            # Current time not in this day's list — snap to closest
            current_hour = _to_local(_from_iso(current_time_iso)).hour
            idx = min(
                range(len(time_isos)),
                key=lambda i: abs(
                    _to_local(_from_iso(time_isos[i])).hour - current_hour
                ),
            )

        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(time_isos):
            return no_update
        return time_isos[new_idx]

    # -------------------------------------------------------------------
    # Update current hour label from selected-time-store
    # -------------------------------------------------------------------
    @app.callback(
        Output("current-hour-label", "children"),
        Input("selected-time-store", "data"),
    )
    def update_hour_label(selected_time_iso):
        if not selected_time_iso:
            return ""
        local_t = _to_local(_from_iso(selected_time_iso))
        return local_t.strftime("%H:%M")

    # -------------------------------------------------------------------
    # Arrow keys: left/right switch day
    # Uses a clientside callback that directly computes the new day
    # from the current day-radio value and the day-keys list.
    # -------------------------------------------------------------------
    app.clientside_callback(
        """
        function(currentDay, dayKeys) {
            // Install the keyboard listener once, updating closure vars
            if (!window._pgw_arrow) {
                window._pgw_arrow = {currentDay: currentDay, dayKeys: dayKeys};
                document.addEventListener('keydown', function(e) {
                    // Escape closes the windgram modal
                    if (e.key === 'Escape') {
                        var closeBtn = document.getElementById('modal-close-btn');
                        if (closeBtn) closeBtn.click();
                        return;
                    }
                    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
                    e.preventDefault();
                    var state = window._pgw_arrow;
                    var idx = state.dayKeys.indexOf(state.currentDay);
                    if (idx < 0) return;
                    var newIdx = e.key === 'ArrowLeft' ? idx - 1 : idx + 1;
                    if (newIdx < 0 || newIdx >= state.dayKeys.length) return;
                    // Programmatically click the correct pill in the main radio
                    var pills = document.querySelectorAll(
                        '#day-radio .dash-options-list-option'
                    );
                    if (pills[newIdx]) pills[newIdx].click();
                });
            }
            // Keep closure in sync with current values
            window._pgw_arrow.currentDay = currentDay;
            window._pgw_arrow.dayKeys = dayKeys;
            return window.dash_clientside.no_update;
        }
        """,
        Output("keyboard-listener", "data-dummy"),
        Input("day-radio", "value"),
        Input("day-keys-store", "data"),
    )

    return app
