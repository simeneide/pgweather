import { useCallback, useEffect, useMemo, useState } from "react";

import { AirgramModal } from "./components/AirgramModal";
import { BrandLogo } from "./components/BrandLogo";
import { ForecastMap } from "./components/ForecastMap";
import { useAirgramSummary } from "./hooks/useAirgramSummary";
import { useFrontendMeta } from "./hooks/useFrontendMeta";
import { useMapPayload } from "./hooks/useMapPayload";

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

/* ------------------------------------------------------------------ */
/* App component                                                      */
/* ------------------------------------------------------------------ */

function App() {
  const {
    loading,
    error: metaError,
    meta,
    modelSource,
    selectedDay,
    selectedTime,
    setSelectedDay,
    setSelectedTime,
    changeModelSource
  } = useFrontendMeta();

  const [selectedName, setSelectedName] = useState("");
  const [takeoffQuery, setTakeoffQuery] = useState("");
  const [showTakeoffResults, setShowTakeoffResults] = useState(false);
  const [windAltitude, setWindAltitude] = useState("off");
  const [modalOpen, setModalOpen] = useState(false);

  const { mapPayload, error: mapError } = useMapPayload({
    selectedTime,
    windAltitude,
    modelSource,
    enabled: !loading
  });

  const {
    airgramPayload,
    summary,
    airgramLoading,
    error: airgramError
  } = useAirgramSummary({
    selectedName,
    selectedDay,
    selectedTime,
    modelSource
  });

  const error = metaError || mapError || airgramError;

  const dayEntry = useMemo(
    () => (meta?.days || []).find((d) => d.key === selectedDay),
    [meta, selectedDay]
  );
  const locationOptions = meta?.location_options || [];
  const modelOptions = meta?.model_source_options || [];

  const selectedLocationOption = useMemo(
    () => locationOptions.find((o) => o.value === selectedName) || null,
    [locationOptions, selectedName]
  );

  const filteredLocationOptions = useMemo(() => {
    const q = takeoffQuery.trim().toLowerCase();
    if (!q) return locationOptions.slice(0, 40);
    return locationOptions
      .filter(
        (o) =>
          o.label.toLowerCase().includes(q) || o.value.toLowerCase().includes(q)
      )
      .slice(0, 40);
  }, [locationOptions, takeoffQuery]);

  useEffect(() => {
    if (!selectedName) {
      setTakeoffQuery("");
      return;
    }
    if (selectedLocationOption) {
      setTakeoffQuery(selectedLocationOption.label);
    }
  }, [selectedLocationOption, selectedName]);

  const commitTakeoffSelection = useCallback(
    (rawValue) => {
      const value = rawValue.trim();
      if (!value) {
        setSelectedName("");
        setModalOpen(false);
        return;
      }

      const lower = value.toLowerCase();
      const exactMatch = locationOptions.find(
        (o) => o.value.toLowerCase() === lower || o.label.toLowerCase() === lower
      );
      const containsMatch =
        exactMatch ||
        locationOptions.find(
          (o) =>
            o.label.toLowerCase().includes(lower) ||
            o.value.toLowerCase().includes(lower)
        );

      if (containsMatch) {
        setSelectedName(containsMatch.value);
        setTakeoffQuery(containsMatch.label);
        setModalOpen(true);
      }
    },
    [locationOptions]
  );

  const handleModelChange = useCallback(async (newModel) => {
    if (newModel === modelSource) return;
    const payload = await changeModelSource(newModel);
    if (!payload) return;
    setSelectedName("");
    setModalOpen(false);
  }, [changeModelSource, modelSource]);

  const handleMapSelectName = useCallback((name) => {
    setSelectedName(name);
    setModalOpen(true);
  }, []);

  /* ---- 5. Escape key ---- */

  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") setModalOpen(false); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  /* ---- Render ---- */

  if (loading) return <div className="page loading-state">Loading...</div>;

  const currentHour = selectedTime
    ? `${String(toLocalHour(selectedTime)).padStart(2, "0")}:00`
    : "";

  return (
    <div className="page">
      <header className="app-header">
        <BrandLogo />
      </header>
      {error ? <div className="error">{error}</div> : null}

      <div className="controls">
        <select
          aria-label="Forecast model"
          value={modelSource}
          onChange={(e) => handleModelChange(e.target.value)}
        >
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

        <div className="takeoff-search">
          <input
            type="text"
            aria-label="Takeoff search"
            placeholder="Search takeoff..."
            value={takeoffQuery}
            onFocus={() => setShowTakeoffResults(true)}
            onBlur={() => {
              window.setTimeout(() => setShowTakeoffResults(false), 120);
            }}
            onChange={(e) => {
              const next = e.target.value;
              setTakeoffQuery(next);
              setShowTakeoffResults(true);
              if (!next.trim()) {
                setSelectedName("");
                setModalOpen(false);
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commitTakeoffSelection(takeoffQuery);
                setShowTakeoffResults(false);
              } else if (e.key === "Escape") {
                setShowTakeoffResults(false);
              }
            }}
          />
          {showTakeoffResults ? (
            <div className="takeoff-results">
              {filteredLocationOptions.length ? (
                filteredLocationOptions.map((o) => (
                  <button
                    type="button"
                    key={o.value}
                    className={selectedName === o.value ? "takeoff-option active" : "takeoff-option"}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      commitTakeoffSelection(o.value);
                      setShowTakeoffResults(false);
                    }}
                  >
                    {o.label}
                  </button>
                ))
              ) : (
                <div className="takeoff-empty">No matching takeoffs</div>
              )}
            </div>
          ) : null}
        </div>

        <select
          aria-label="Wind altitude"
          value={windAltitude}
          onChange={(e) => setWindAltitude(e.target.value)}
        >
          <option value="off">Wind off</option>
          <option value="0">Surface wind</option>
          <option value="500">Wind 500m</option>
          <option value="1000">Wind 1000m</option>
          <option value="1500">Wind 1500m</option>
          <option value="2000">Wind 2000m</option>
          <option value="3000">Wind 3000m</option>
        </select>
      </div>

      <div className="map-shell">
        <ForecastMap
          mapPayload={mapPayload}
          selectedName={selectedName}
          onSelectName={handleMapSelectName}
        />
      </div>
      {summary ? (
        <div className="summary" onClick={() => setModalOpen(true)} role="button" tabIndex={0}>
          {summary}
        </div>
      ) : null}

      <AirgramModal
        open={modalOpen}
        selectedName={selectedName}
        days={meta?.days || []}
        selectedDay={selectedDay}
        selectedTime={selectedTime}
        onSelectDay={(dayKey, nextTime) => {
          setSelectedDay(dayKey);
          if (nextTime) setSelectedTime(nextTime);
        }}
        summary={summary}
        airgramLoading={airgramLoading}
        airgramPayload={airgramPayload}
        onClose={() => setModalOpen(false)}
        toLocalHour={toLocalHour}
        nearestHourIso={nearestHourIso}
      />

      <footer className="app-footer">
        <div>Hosted and maintained by <a href="https://eide.ai">eide.ai</a></div>
        <div>
          Data: <a href="https://www.met.no/">MET Norway / MEPS</a> | Weather symbols: <a href="https://www.yr.no/">Yr</a> | Map: <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
