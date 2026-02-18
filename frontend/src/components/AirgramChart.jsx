import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

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
    zmin: 0,
    zmax: 5,
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
    type: "scatter",
    mode: "lines",
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
    type: "scatter",
    mode: "markers+text",
    x: payload.wind_samples.map((s) => s.time_label),
    y: payload.wind_samples.map((s) => s.altitude),
    text: payload.wind_samples.map((s) => `${Math.round(s.wind_speed)}`),
    textposition: "middle right",
    textfont: { size: 9, color: "#555", family: "Arial" },
    cliponaxis: false,
    marker: {
      symbol: "arrow",
      size: 12,
      angle: payload.wind_samples.map((s) => s.wind_direction),
      color: payload.wind_samples.map((s) => windColor(s.wind_speed)),
      line: { width: 0.5, color: "rgba(0,0,0,0.3)" }
    },
    hoverinfo: "text",
    hovertext: payload.wind_samples.map(
      (s) => `Alt: ${s.altitude.toFixed(0)}m | Wind: ${s.wind_speed.toFixed(0)} m/s | ${s.wind_direction.toFixed(0)}\u00b0 | Climb: ${s.thermal_velocity.toFixed(1)} m/s`
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
      type: "rect",
      xref: "paper",
      yref: "y",
      x0: 0,
      x1: 1,
      y0: 0,
      y1: elevation,
      fillcolor: "rgba(180,180,180,0.7)",
      line: { width: 0 },
      layer: "above"
    });
  }
  if (idx >= 0) {
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "y",
      x0: idx - 0.5,
      x1: idx + 0.5,
      y0: 0,
      y1: payload.altitude_max,
      fillcolor: "rgba(59,130,246,0.10)",
      line: { color: "rgba(59,130,246,0.4)", width: 1.2 }
    });
  }

  const annotations = [];
  if (payload.snow_depth_cm != null && payload.snow_depth_cm > 0) {
    const cm = payload.snow_depth_cm;
    annotations.push({
      x: 0.01,
      y: 0.01,
      xref: "paper",
      yref: "paper",
      text: `<b>\u2744\ufe0f ${cm >= 1 ? `${cm.toFixed(0)} cm snow` : "<1 cm snow"}</b>`,
      showarrow: false,
      font: { size: 12, color: "#4a90d9" },
      bgcolor: "rgba(255,255,255,0.8)",
      borderpad: 3,
      xanchor: "left",
      yanchor: "bottom"
    });
  }

  Plotly.newPlot(el, [heatmap, topLine, windScatter], {
    title: `${payload.location} - ${payload.date}`,
    height: 460,
    margin: { l: 48, r: 12, t: 56, b: 20 },
    xaxis: { type: "category", categoryorder: "array", categoryarray: zCols, side: "top", fixedrange: true },
    yaxis: { range: [0, payload.altitude_max], ticksuffix: "m", fixedrange: true },
    shapes,
    annotations
  }, { displayModeBar: false, responsive: true });
}

export function AirgramChart({ payload, hidden = false }) {
  const chartRef = useRef(null);

  useEffect(() => {
    renderWindgram(chartRef.current, payload);
  }, [payload]);

  if (hidden) {
    return <div ref={chartRef} style={{ display: "none" }} />;
  }
  return <div ref={chartRef} className="airgram" />;
}
