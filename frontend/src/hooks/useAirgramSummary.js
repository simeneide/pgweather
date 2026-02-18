import { useEffect, useRef, useState } from "react";

import { postJson } from "../lib/api";

function toLocalHour(iso) {
  return Number(
    new Intl.DateTimeFormat("en-GB", {
      hour: "2-digit",
      hourCycle: "h23",
      timeZone: "Europe/Oslo"
    }).format(new Date(iso))
  );
}

export function useAirgramSummary({ selectedName, selectedDay, selectedTime, modelSource }) {
  const [airgramPayload, setAirgramPayload] = useState(null);
  const [summary, setSummary] = useState("");
  const [airgramLoading, setAirgramLoading] = useState(false);
  const [error, setError] = useState("");
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (!selectedName || !selectedDay || !selectedTime) {
      setAirgramPayload(null);
      setSummary("");
      setAirgramLoading(false);
      return;
    }

    setAirgramPayload(null);
    setSummary("");
    setAirgramLoading(true);
    setError("");

    const abortController = new AbortController();
    const reqId = ++requestIdRef.current;

    Promise.all([
      postJson(
        "/api/frontend/airgram",
        {
          location: selectedName,
          selected_date: selectedDay,
          altitude_max: 3000,
          selected_hour: toLocalHour(selectedTime),
          model_source: modelSource
        },
        { signal: abortController.signal }
      ),
      postJson(
        "/api/frontend/summary",
        {
          selected_name: selectedName,
          selected_time: selectedTime,
          model_source: modelSource
        },
        { signal: abortController.signal }
      )
    ])
      .then(([airgram, summaryPayload]) => {
        if (abortController.signal.aborted) return;
        if (reqId !== requestIdRef.current) return;
        setAirgramPayload(airgram);
        setSummary(
          `${summaryPayload.summary} | Updated ${new Date(summaryPayload.forecast_used_timestamp).toLocaleString("nb-NO", { timeZone: "Europe/Oslo" })} local (${summaryPayload.forecast_age_hours.toFixed(1)}h ago)`
        );
      })
      .catch((e) => {
        if (abortController.signal.aborted) return;
        if (reqId !== requestIdRef.current) return;
        setError(e.message);
      })
      .finally(() => {
        if (abortController.signal.aborted) return;
        if (reqId !== requestIdRef.current) return;
        setAirgramLoading(false);
      });

    return () => {
      abortController.abort();
    };
  }, [selectedName, selectedDay, selectedTime, modelSource]);

  return { airgramPayload, summary, airgramLoading, error, setSummary, setAirgramPayload };
}
