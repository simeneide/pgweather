export function pickNearestFeatureName(features, projectLngLat, clickPoint) {
  let bestName = null;
  let bestDistSq = Number.POSITIVE_INFINITY;

  for (const f of features) {
    if (f.geometry?.type !== "Point") continue;
    const coords = f.geometry.coordinates;
    if (!Array.isArray(coords) || coords.length < 2) continue;
    const projected = projectLngLat({ lng: coords[0], lat: coords[1] });
    const dx = projected.x - clickPoint.x;
    const dy = projected.y - clickPoint.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDistSq) {
      bestDistSq = d2;
      bestName = f.properties?.name;
    }
  }

  return typeof bestName === "string" ? bestName : null;
}
