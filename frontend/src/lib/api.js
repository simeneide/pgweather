export function postJson(url, payload, options = {}) {
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
