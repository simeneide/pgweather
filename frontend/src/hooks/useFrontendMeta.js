import { useCallback, useEffect, useRef, useState } from "react";

import { postJson } from "../lib/api";

export function useFrontendMeta() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [meta, setMeta] = useState(null);
  const [modelSource, setModelSource] = useState("");
  const [selectedDay, setSelectedDay] = useState("");
  const [selectedTime, setSelectedTime] = useState("");

  const requestIdRef = useRef(0);

  const applyMetaPayload = useCallback((payload) => {
    setMeta(payload);
    setModelSource(payload.selected_model_source);
    setSelectedDay(payload.selected_day);
    setSelectedTime(payload.selected_time);
  }, []);

  const fetchMeta = useCallback(async (nextModelSource) => {
    const reqId = ++requestIdRef.current;
    const payload = nextModelSource ? { model_source: nextModelSource } : {};
    const result = await postJson("/api/frontend/meta", payload);
    if (reqId !== requestIdRef.current) return null;
    applyMetaPayload(result);
    return result;
  }, [applyMetaPayload]);

  useEffect(() => {
    let mounted = true;
    fetchMeta(null)
      .catch((e) => {
        if (mounted) setError(e.message);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
      requestIdRef.current += 1;
    };
  }, [fetchMeta]);

  const changeModelSource = useCallback(async (nextModelSource) => {
    if (nextModelSource === modelSource) return null;
    setError("");
    try {
      return await fetchMeta(nextModelSource);
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [fetchMeta, modelSource]);

  return {
    loading,
    error,
    meta,
    modelSource,
    selectedDay,
    selectedTime,
    setSelectedDay,
    setSelectedTime,
    changeModelSource
  };
}
