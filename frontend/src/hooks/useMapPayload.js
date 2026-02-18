import { useEffect, useRef, useState } from "react";

import { postJson } from "../lib/api";

export function useMapPayload({ selectedTime, windAltitude, modelSource, enabled }) {
  const [mapPayload, setMapPayload] = useState(null);
  const [error, setError] = useState("");
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (!enabled || !selectedTime || !modelSource) return;
    const abortController = new AbortController();
    const reqId = ++requestIdRef.current;

    postJson(
      "/api/frontend/map",
      {
        selected_time: selectedTime,
        selected_name: null,
        zoom: 6,
        wind_altitude: windAltitude === "off" ? null : Number(windAltitude),
        model_source: modelSource
      },
      { signal: abortController.signal }
    )
      .then((payload) => {
        if (reqId !== requestIdRef.current) return;
        setMapPayload(payload);
      })
      .catch((e) => {
        if (abortController.signal.aborted) return;
        if (reqId !== requestIdRef.current) return;
        setError(e.message);
      });

    return () => {
      abortController.abort();
    };
  }, [enabled, selectedTime, windAltitude, modelSource]);

  return { mapPayload, error };
}
