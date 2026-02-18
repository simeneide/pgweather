import { AirgramChart } from "./AirgramChart";

export function AirgramModal({
  open,
  selectedName,
  days,
  selectedDay,
  selectedTime,
  onSelectDay,
  summary,
  airgramLoading,
  airgramPayload,
  onClose,
  toLocalHour,
  nearestHourIso
}) {
  if (!open || !selectedName) {
    return <AirgramChart payload={airgramPayload} hidden />;
  }

  return (
    <div className="modal-wrapper">
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal-content">
        <button className="modal-close-btn" onClick={onClose}>&times;</button>
        <div className="modal-header">
          <h2>{selectedName}</h2>
          <div className="modal-day-row">
            {(days || []).map((day) => (
              <button
                key={day.key}
                className={selectedDay === day.key ? "pill active" : "pill"}
                onClick={() => {
                  const next = nearestHourIso(day.times, toLocalHour(selectedTime));
                  onSelectDay(day.key, next);
                }}
              >
                {day.label}
              </button>
            ))}
          </div>
          {summary ? <div className="modal-summary">{summary}</div> : null}
        </div>
        {airgramLoading ? (
          <div className="airgram" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
            Loading windgram...
          </div>
        ) : (
          <AirgramChart payload={airgramPayload} />
        )}
      </div>
    </div>
  );
}
