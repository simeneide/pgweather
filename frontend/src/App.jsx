import { useCallback, useEffect, useMemo, useState } from "react";

import { AirgramModal } from "./components/AirgramModal";
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
  const [windAltitude, setWindAltitude] = useState("off");
  const [modalOpen, setModalOpen] = useState(false);
  const [mapColorMode, setMapColorMode] = useState("suitability");

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

      <ForecastMap
        mapPayload={mapPayload}
        selectedName={selectedName}
        mapColorMode={mapColorMode}
        onSelectName={handleMapSelectName}
      />
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
