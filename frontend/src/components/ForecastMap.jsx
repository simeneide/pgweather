import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";

import { pickNearestFeatureName } from "../lib/nearest";

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

export function ForecastMap({ mapPayload, selectedName, mapColorMode, onSelectName }) {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const [mapInitError, setMapInitError] = useState("");

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    let map;
    try {
      map = new maplibregl.Map({
        container: mapContainerRef.current,
        style: OSM_STYLE,
        center: [8, 61.2],
        zoom: 6
      });
    } catch {
      setMapInitError("Unable to initialize map (WebGL not available).");
      return;
    }

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    const popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      offset: 12,
      maxWidth: "280px"
    });

    map.on("mouseenter", "points-fill", () => {
      map.getCanvas().style.cursor = "pointer";
    });
    map.on("mouseleave", "points-fill", () => {
      map.getCanvas().style.cursor = "";
      popup.remove();
    });
    map.on("mousemove", "points-fill", (e) => {
      if (!e.features?.length) {
        popup.remove();
        return;
      }
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

      const bestName = pickNearestFeatureName(
        features,
        (lngLat) => map.project(lngLat),
        e.point
      );

      if (typeof bestName === "string") onSelectName(bestName);
    });

    mapRef.current = map;
    return () => {
      popup.remove();
      map.remove();
      mapRef.current = null;
    };
  }, [onSelectName]);

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

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    map.setPaintProperty(
      "points-fill",
      "circle-color",
      mapColorMode === "suitability" ? SUITABILITY_COLOR : THERMAL_COLOR
    );
  }, [mapColorMode]);

  const legend = useMemo(() => {
    if (mapColorMode === "suitability") {
      return (
        <>
          <span className="legend-dot" style={{ background: "#4caf50" }} /> Suitable{" "}
          <span className="legend-dot" style={{ background: "#ff9800" }} /> Marginal{" "}
          <span className="legend-dot" style={{ background: "#f44336" }} /> Not suitable{" "}
          <span className="legend-dot" style={{ background: "#9e9e9e" }} /> No data
        </>
      );
    }
    return (
      <>
        <span className="legend-dot" style={{ background: "#d9d9d9" }} /> 0{" "}
        <span className="legend-dot" style={{ background: "#ffd75a" }} /> 1.5{" "}
        <span className="legend-dot" style={{ background: "#ff9f35" }} /> 2.5{" "}
        <span className="legend-dot" style={{ background: "#ef5a2f" }} /> 3.5{" "}
        <span className="legend-dot" style={{ background: "#b91c1c" }} /> 5+ m/s
      </>
    );
  }, [mapColorMode]);

  return (
    <div style={{ position: "relative" }}>
      {mapInitError ? <div className="error">{mapInitError}</div> : null}
      <div ref={mapContainerRef} className="map" />
      <div className="map-legend">{legend}</div>
    </div>
  );
}
