import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import Plotly from "plotly.js-dist-min";

const EMPTY_FC = { type: "FeatureCollection", features: [] };

const THERMAL_COLOR = [
  "interpolate",
  ["linear"],
  ["get", "peak_thermal_velocity"],
  0, "#d9d9d9",
  0.5, "#fff6a4",
  1.5, "#ffd75a",
  2.5, "#ff9f35",
  3.5, "#ef5a2f",
  5, "#b91c1c"
];

const SUITABILITY_COLOR = ["get", "suitability_color"];

const OSM_STYLE = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: [
        "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"
      ],
      tileSize: 256,
      attribution: "OpenStreetMap contributors"
    },
    areas: { type: "geojson", data: EMPTY_FC },
    points: { type: "geojson", data: EMPTY_FC },
    wind: { type: "geojson", data: EMPTY_FC }
  },
  layers: [
    { id: "osm", type: "raster", source: "osm" },
    {
      id: "areas-fill",
      type: "fill",
      source: "areas",
      paint: { "fill-color": THERMAL_COLOR, "fill-opacity": 0.3 }
    },
    {
      id: "areas-line",
      type: "line",
      source: "areas",
      paint: { "line-color": "rgba(71,85,105,0.4)", "line-width": 0.8 }
    },
    {
      id: "wind-lines",
      type: "line",
      source: "wind",
      paint: { "line-width": 1.4, "line-color": "rgba(30,41,59,0.5)" }
    },
    {
      id: "points-outline",
      type: "circle",
      source: "points",
      paint: {
        "circle-radius": ["case", ["==", ["get", "selected"], 1], 14, 9],
        "circle-color": "rgba(15,23,42,0.6)"
      }
    },
    {
      id: "points-fill",
      type: "circle",
      source: "points",
      paint: {
        "circle-radius": ["case", ["==", ["get", "selected"], 1], 10, 6],
        "circle-color": THERMAL_COLOR
      }
    }
  ]
};

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

function toLocalHour(iso) {
  return Number(
    new Intl.DateTimeFormat("en-GB", {
      hour: "2-digit",
      hourCycle: "h23",
      timeZone: "Europe/Oslo"
    }).format(new Date(iso))
  );
}

function nearestHourIso(times, targetHour) {
  if (!times.length) return null;
  let best = times[0];
  let bestDiff = 99;
  for (const iso of times) {
    const diff = Math.abs(toLocalHour(iso) - targetHour);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = iso;
    }
  }
  return best;
}

function postJson(url, payload, options = {}) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: options.signal
  }).then((r) => {
    if (!r.ok) throw new Error(`${url} failed: ${r.status}`);
    return r.json();
  });
}

/** Build GeoJSON for the map. `selectedName` is applied here (frontend-only). */
function buildGeoJson(mapPayload, selectedName) {
  return {
    points: {
      type: "FeatureCollection",
      features: mapPayload.points.map((p) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [p.longitude, p.latitude] },
        properties: {
          name: p.name,
          thermal_top: p.thermal_top,
          peak_thermal_velocity: p.peak_thermal_velocity,
          selected: p.name === selectedName ? 1 : 0,
          suitability_color: p.suitability_color,
          suitability_label: p.suitability_label,
          suitability_tooltip: p.suitability_tooltip,
          wind_speed: p.wind_speed,
          wind_direction_compass: p.wind_direction_compass
        }
      }))
    },
    areas: { type: "FeatureCollection", features: mapPayload.area_features },
    wind: {
      type: "FeatureCollection",
      features: mapPayload.wind_vectors.map((v) => ({
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: [
            [v.longitude, v.latitude],
            [v.tip_longitude, v.tip_latitude]
          ]
        },
        properties: { wind_speed: v.wind_speed }
      }))
    }
  };
}

/* ------------------------------------------------------------------ */
/* Windgram chart builder (extracted so it's not inline in JSX)       */
/* ------------------------------------------------------------------ */

function renderWindgram(el, payload) {
  if (!el) return;
  if (!payload || !payload.time_labels.length) {
    Plotly.newPlot(el, [], {
      title: "Select a takeoff on the map to see the windgram",
      height: 350,
      margin: { l: 40, r: 12, t: 48, b: 24 }
    }, { displayModeBar: false, responsive: true });
    return;
  }

  const zCols = payload.time_labels;
  const heatmap = {
    type: "heatmap",
    z: payload.thermal_matrix,
    x: zCols,
    y: payload.altitudes,
    zmin: 0, zmax: 5,
    colorscale: [
      [0.0, "rgb(255,255,255)"],
      [0.02, "rgb(255,255,210)"],
      [0.2, "rgb(255,245,80)"],
      [0.4, "rgb(255,220,50)"],
      [0.6, "rgb(255,180,30)"],
      [0.8, "rgb(255,140,0)"],
      [1.0, "rgb(255,80,0)"]
    ],
    showscale: false,
    hovertemplate: "Alt: %{y:.0f}m<br>Time: %{x}<br>Climb: %{z:.1f} m/s<extra></extra>"
  };

  const topLine = {
    type: "scatter", mode: "lines",
    x: payload.thermal_tops.map((p) => p.time_label),
    y: payload.thermal_tops.map((p) => p.thermal_top),
    line: { color: "rgba(180,80,0,0.7)", width: 2, dash: "dot" },
    hovertemplate: "Thermal top: %{y:.0f}m<extra></extra>",
    showlegend: false
  };

  const windColor = (spd) => {
    if (spd < 2) return "#b0b0b0";
    if (spd < 4) return "#4caf50";
    if (spd < 6) return "#ffeb3b";
    if (spd < 8) return "#ff9800";
    if (spd < 12) return "#f44336";
    return "#4a148c";
  };

  const windScatter = {
    type: "scatter", mode: "markers+text",
    x: payload.wind_samples.map((s) => s.time_label),
    y: payload.wind_samples.map((s) => s.altitude),
    text: payload.wind_samples.map((s) => `${Math.round(s.wind_speed)}`),
    textposition: "middle right",
    textfont: { size: 9, color: "#555", family: "Arial" },
    cliponaxis: false,
    marker: {
      symbol: "arrow", size: 12,
      angle: payload.wind_samples.map((s) => s.wind_direction),
      color: payload.wind_samples.map((s) => windColor(s.wind_speed)),
      line: { width: 0.5, color: "rgba(0,0,0,0.3)" }
    },
    hoverinfo: "text",
    hovertext: payload.wind_samples.map((s) =>
      `Alt: ${s.altitude.toFixed(0)}m | Wind: ${s.wind_speed.toFixed(0)} m/s | ${s.wind_direction.toFixed(0)}\u00b0 | Climb: ${s.thermal_velocity.toFixed(1)} m/s`
    ),
    showlegend: false
  };

  const selLabel = payload.selected_hour == null
    ? null
    : `${String(payload.selected_hour).padStart(2, "0")}h`;
  const idx = selLabel ? zCols.indexOf(selLabel) : -1;
  const elevation = payload.elevation || 0;
  const shapes = [];

  if (elevation > 0) {
    shapes.push({
      type: "rect", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: 0, y1: elevation,
      fillcolor: "rgba(180,180,180,0.7)", line: { width: 0 }, layer: "above"
    });
  }
  if (idx >= 0) {
    shapes.push({
      type: "rect", xref: "x", yref: "y",
      x0: idx - 0.5, x1: idx + 0.5, y0: 0, y1: payload.altitude_max,
      fillcolor: "rgba(59,130,246,0.10)",
      line: { color: "rgba(59,130,246,0.4)", width: 1.2 }
    });
  }

  const annotations = [];
  if (payload.snow_depth_cm != null && payload.snow_depth_cm > 0) {
    const cm = payload.snow_depth_cm;
    annotations.push({
      x: 0.01, y: 0.01, xref: "paper", yref: "paper",
      text: `<b>\u2744\ufe0f ${cm >= 1 ? `${cm.toFixed(0)} cm snow` : "<1 cm snow"}</b>`,
      showarrow: false,
      font: { size: 12, color: "#4a90d9" },
      bgcolor: "rgba(255,255,255,0.8)", borderpad: 3,
      xanchor: "left", yanchor: "bottom"
    });
  }

  Plotly.newPlot(el, [heatmap, topLine, windScatter], {
    title: `${payload.location} - ${payload.date}`,
    height: 460,
    margin: { l: 48, r: 12, t: 56, b: 20 },
    xaxis: { type: "category", categoryorder: "array", categoryarray: zCols, side: "top", fixedrange: true },
    yaxis: { range: [0, payload.altitude_max], ticksuffix: "m", fixedrange: true },
    shapes, annotations
  }, { displayModeBar: false, responsive: true });
}

/* ------------------------------------------------------------------ */
/* App component                                                      */
/* ------------------------------------------------------------------ */

function App() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const chartRef = useRef(null);
  // Track whether the initial meta fetch has populated modelSource
  const initializedRef = useRef(false);
  const metaRequestIdRef = useRef(0);
  const mapRequestIdRef = useRef(0);
  const airgramRequestIdRef = useRef(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [meta, setMeta] = useState(null);
  const [modelSource, setModelSource] = useState("");
  const [selectedDay, setSelectedDay] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const [windAltitude, setWindAltitude] = useState("off");
  const [mapPayload, setMapPayload] = useState(null);
  const [airgramPayload, setAirgramPayload] = useState(null);
  const [airgramLoading, setAirgramLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [mapColorMode, setMapColorMode] = useState("suitability");

  const dayEntry = useMemo(
    () => (meta?.days || []).find((d) => d.key === selectedDay),
    [meta, selectedDay]
  );
  const locationOptions = meta?.location_options || [];
  const modelOptions = meta?.model_source_options || [];

  /* ---- Fetch helpers ---- */

  const fetchMeta = useCallback((ms) => {
    return postJson("/api/frontend/meta", ms ? { model_source: ms } : {});
  }, []);

  /* ---- 1. Initial meta fetch (once) ---- */

  useEffect(() => {
    const reqId = ++metaRequestIdRef.current;
    fetchMeta(null)
      .then((p) => {
        if (reqId !== metaRequestIdRef.current) return;
        setMeta(p);
        setModelSource(p.selected_model_source);
        setSelectedDay(p.selected_day);
        setSelectedTime(p.selected_time);
        initializedRef.current = true;
      })
      .catch((e) => {
        if (reqId === metaRequestIdRef.current) setError(e.message);
      })
      .finally(() => {
        if (reqId === metaRequestIdRef.current) setLoading(false);
      });
  }, [fetchMeta]);

  /* ---- 2. Model source change (user-initiated only) ---- */

  const handleModelChange = useCallback((newModel) => {
    if (newModel === modelSource) return;
    setModelSource(newModel);
    const reqId = ++metaRequestIdRef.current;
    fetchMeta(newModel)
      .then((p) => {
        if (reqId !== metaRequestIdRef.current) return;
        setMeta(p);
        setSelectedDay(p.selected_day);
        setSelectedTime(p.selected_time);
        setSelectedName("");
        setAirgramPayload(null);
        setSummary("");
      })
      .catch((e) => {
        if (reqId === metaRequestIdRef.current) setError(e.message);
      });
  }, [modelSource, fetchMeta]);

  /* ---- 3. Map payload fetch — only when time/wind/model change ---- */

  useEffect(() => {
    if (!selectedTime || !modelSource || !initializedRef.current) return;
    const reqId = ++mapRequestIdRef.current;
    postJson("/api/frontend/map", {
      selected_time: selectedTime,
      selected_name: null,
      zoom: 6,
      wind_altitude: windAltitude === "off" ? null : Number(windAltitude),
      model_source: modelSource
    })
      .then((p) => {
        if (reqId === mapRequestIdRef.current) setMapPayload(p);
      })
      .catch((e) => {
        if (reqId === mapRequestIdRef.current) setError(e.message);
      });
  }, [selectedTime, windAltitude, modelSource]);

  /* ---- 4. Airgram + summary fetch ---- */

  useEffect(() => {
    if (!selectedName || !selectedDay || !selectedTime) {
      setAirgramPayload(null);
      setSummary("");
      setAirgramLoading(false);
      return;
    }

    // Clear immediately so user doesn't see stale chart
    setAirgramPayload(null);
    setSummary("");
    setAirgramLoading(true);

    const abortController = new AbortController();
    const reqId = ++airgramRequestIdRef.current;

    Promise.all([
      postJson("/api/frontend/airgram", {
        location: selectedName,
        selected_date: selectedDay,
        altitude_max: 3000,
        selected_hour: toLocalHour(selectedTime),
        model_source: modelSource
      }, { signal: abortController.signal }),
      postJson("/api/frontend/summary", {
        selected_name: selectedName,
        selected_time: selectedTime,
        model_source: modelSource
      }, { signal: abortController.signal })
    ])
      .then(([airgram, sum]) => {
        if (abortController.signal.aborted) return;
        if (reqId !== airgramRequestIdRef.current) return;
        setAirgramPayload(airgram);
        setSummary(
          `${sum.summary} | Updated ${new Date(sum.forecast_used_timestamp).toLocaleString("nb-NO", { timeZone: "Europe/Oslo" })} local (${sum.forecast_age_hours.toFixed(1)}h ago)`
        );
      })
      .catch((e) => {
        if (!abortController.signal.aborted && reqId === airgramRequestIdRef.current) {
          setError(e.message);
        }
      })
      .finally(() => {
        if (!abortController.signal.aborted && reqId === airgramRequestIdRef.current) {
          setAirgramLoading(false);
        }
      });

    return () => abortController.abort();
  }, [selectedName, selectedDay, selectedTime, modelSource]);

  /* ---- 5. Map initialization (once, after loading) ---- */

  useEffect(() => {
    if (loading || !mapContainerRef.current || mapRef.current) return;

    let map;
    try {
      map = new maplibregl.Map({
        container: mapContainerRef.current,
        style: OSM_STYLE,
        center: [8, 61.2],
        zoom: 6
      });
    } catch {
      setError("Unable to initialize map (WebGL not available).");
      return;
    }

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    const popup = new maplibregl.Popup({
      closeButton: false, closeOnClick: false, offset: 12, maxWidth: "280px"
    });

    map.on("mouseenter", "points-fill", () => {
      map.getCanvas().style.cursor = "pointer";
    });
    map.on("mouseleave", "points-fill", () => {
      map.getCanvas().style.cursor = "";
      popup.remove();
    });
    map.on("mousemove", "points-fill", (e) => {
      if (!e.features?.length) { popup.remove(); return; }
      const props = e.features[0].properties;
      const name = props.name || "";
      const thermalTop = Math.round(props.thermal_top || 0);
      const peakVel = (props.peak_thermal_velocity || 0).toFixed(1);
      const suitLabel = props.suitability_label || "";
      const suitColor = props.suitability_color || "#9e9e9e";
      const windSpd = (props.wind_speed || 0).toFixed(0);
      const windDir = props.wind_direction_compass || "";
      const suitTip = props.suitability_tooltip || "";
      popup.setLngLat(e.lngLat).setHTML(`<div style="font-size:13px;line-height:1.4">
        <strong>${name}</strong><br>
        <span style="color:${suitColor};font-weight:700">${suitLabel}</span>
        ${suitTip ? `<br><span style="color:#6b7280;font-size:11px">${suitTip}</span>` : ""}
        <br>Thermal: ${peakVel} m/s | Top: ${thermalTop}m
        <br>Wind: ${windDir} ${windSpd} m/s
      </div>`).addTo(map);
    });

    map.on("click", (e) => {
      const tolerance = 12;
      const bbox = [
        [e.point.x - tolerance, e.point.y - tolerance],
        [e.point.x + tolerance, e.point.y + tolerance]
      ];
      const features = map.queryRenderedFeatures(bbox, { layers: ["points-fill"] });
      if (!features.length) return;

      // Pick the closest marker to click point (not just first rendered feature)
      let bestName = null;
      let bestDistSq = Number.POSITIVE_INFINITY;
      for (const f of features) {
        if (f.geometry?.type !== "Point") continue;
        const coords = f.geometry.coordinates;
        if (!Array.isArray(coords) || coords.length < 2) continue;
        const projected = map.project({ lng: coords[0], lat: coords[1] });
        const dx = projected.x - e.point.x;
        const dy = projected.y - e.point.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDistSq) {
          bestDistSq = d2;
          bestName = f.properties?.name;
        }
      }

      if (typeof bestName === "string") {
        setSelectedName(bestName);
        setModalOpen(true);
      }
    });

    mapRef.current = map;
    return () => {
      popup.remove();
      map.remove();
      mapRef.current = null;
    };
  }, [loading]);

  /* ---- 6. Push data to map when payload OR selection changes ---- */
  /*    No API call — just re-build GeoJSON with the new selected flag */

  useEffect(() => {
    if (!mapPayload) return;
    const map = mapRef.current;
    if (!map) return;

    const apply = () => {
      const { points, areas, wind } = buildGeoJson(mapPayload, selectedName);
      map.getSource("areas")?.setData(areas);
      map.getSource("points")?.setData(points);
      map.getSource("wind")?.setData(wind);
    };

    if (map.isStyleLoaded()) {
      apply();
    } else {
      map.once("load", apply);
    }
  }, [mapPayload, selectedName]);

  /* ---- 7. Color mode toggle ---- */

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    map.setPaintProperty("points-fill", "circle-color",
      mapColorMode === "suitability" ? SUITABILITY_COLOR : THERMAL_COLOR);
  }, [mapColorMode]);

  /* ---- 8. Render windgram chart ---- */

  useEffect(() => {
    renderWindgram(chartRef.current, airgramPayload);
  }, [airgramPayload]);

  /* ---- 9. Escape key ---- */

  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") setModalOpen(false); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  /* ---- Render ---- */

  if (loading) return <div className="page">Loading...</div>;

  const currentHour = selectedTime
    ? `${String(toLocalHour(selectedTime)).padStart(2, "0")}:00`
    : "";

  return (
    <div className="page">
      <h1>Termikkvarselet</h1>
      {error ? <div className="error">{error}</div> : null}

      <div className="controls">
        <select value={modelSource} onChange={(e) => handleModelChange(e.target.value)}>
          {modelOptions.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>

        <div className="day-row">
          {(meta?.days || []).map((day) => (
            <button key={day.key}
              className={selectedDay === day.key ? "pill active" : "pill"}
              onClick={() => {
                const next = nearestHourIso(day.times, toLocalHour(selectedTime));
                setSelectedDay(day.key);
                if (next) setSelectedTime(next);
              }}>{day.label}</button>
          ))}
        </div>

        <div className="hour-row">
          <button onClick={() => {
            const t = dayEntry?.times || [];
            const i = t.indexOf(selectedTime);
            if (i > 0) setSelectedTime(t[i - 1]);
          }}>&#9664;</button>
          <span>{currentHour}</span>
          <button onClick={() => {
            const t = dayEntry?.times || [];
            const i = t.indexOf(selectedTime);
            if (i >= 0 && i < t.length - 1) setSelectedTime(t[i + 1]);
          }}>&#9654;</button>
        </div>

        <select value={selectedName} onChange={(e) => {
          setSelectedName(e.target.value);
          if (e.target.value) setModalOpen(true);
        }}>
          <option value="">Search takeoff...</option>
          {locationOptions.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>

        <select value={windAltitude} onChange={(e) => setWindAltitude(e.target.value)}>
          <option value="off">Wind off</option>
          <option value="0">Surface wind</option>
          <option value="500">Wind 500m</option>
          <option value="1000">Wind 1000m</option>
          <option value="1500">Wind 1500m</option>
          <option value="2000">Wind 2000m</option>
          <option value="3000">Wind 3000m</option>
        </select>

        <div className="color-toggle">
          <button className={mapColorMode === "suitability" ? "pill active" : "pill"}
            onClick={() => setMapColorMode("suitability")}>Wind suitability</button>
          <button className={mapColorMode === "thermal" ? "pill active" : "pill"}
            onClick={() => setMapColorMode("thermal")}>Thermals</button>
        </div>
      </div>

      <div style={{ position: "relative" }}>
        <div ref={mapContainerRef} className="map" />
        <div className="map-legend">
          {mapColorMode === "suitability" ? (
            <>
              <span className="legend-dot" style={{ background: "#4caf50" }} /> Suitable{" "}
              <span className="legend-dot" style={{ background: "#ff9800" }} /> Marginal{" "}
              <span className="legend-dot" style={{ background: "#f44336" }} /> Not suitable{" "}
              <span className="legend-dot" style={{ background: "#9e9e9e" }} /> No data
            </>
          ) : (
            <>
              <span className="legend-dot" style={{ background: "#d9d9d9" }} /> 0{" "}
              <span className="legend-dot" style={{ background: "#ffd75a" }} /> 1.5{" "}
              <span className="legend-dot" style={{ background: "#ff9f35" }} /> 2.5{" "}
              <span className="legend-dot" style={{ background: "#ef5a2f" }} /> 3.5{" "}
              <span className="legend-dot" style={{ background: "#b91c1c" }} /> 5+ m/s
            </>
          )}
        </div>
      </div>
      {summary ? (
        <div className="summary" onClick={() => setModalOpen(true)} role="button" tabIndex={0}>
          {summary}
        </div>
      ) : null}

      {modalOpen && selectedName ? (
        <div className="modal-wrapper">
          <div className="modal-backdrop" onClick={() => setModalOpen(false)} />
          <div className="modal-content">
            <button className="modal-close-btn" onClick={() => setModalOpen(false)}>&times;</button>
            <div className="modal-header">
              <h2>{selectedName}</h2>
              <div className="modal-day-row">
                {(meta?.days || []).map((day) => (
                  <button key={day.key}
                    className={selectedDay === day.key ? "pill active" : "pill"}
                    onClick={() => {
                      const next = nearestHourIso(day.times, toLocalHour(selectedTime));
                      setSelectedDay(day.key);
                      if (next) setSelectedTime(next);
                    }}>{day.label}</button>
                ))}
              </div>
              {summary ? <div className="modal-summary">{summary}</div> : null}
            </div>
            {airgramLoading ? (
              <div className="airgram" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                Loading windgram...
              </div>
            ) : (
              <div ref={chartRef} className="airgram" />
            )}
          </div>
        </div>
      ) : (
        <div ref={chartRef} style={{ display: "none" }} />
      )}

      <footer>
        <div>Hosted and maintained by <a href="https://eide.ai">eide.ai</a></div>
        <div>
          Data: <a href="https://www.met.no/">MET Norway / MEPS</a> | Weather symbols: <a href="https://www.yr.no/">Yr</a> | Map: <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
